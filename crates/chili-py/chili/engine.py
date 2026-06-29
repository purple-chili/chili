"""Python bindings for Chili's ``EngineState`` (Rust ``chili-core``)."""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import polars as pl

from .engine_state import EngineState  # type: ignore


class ChiliEngine:
    """High-level Python interface to the Chili evaluation engine.

    Wraps the Rust ``EngineState`` exposed via PyO3 and provides a
    Pythonic API for evaluating Chili/Pepper expressions, managing
    variables, partitioned DataFrames, and IPC connections.

    Args:
        debug: Enable debug-level logging inside the engine.
        lazy: Enable lazy evaluation mode.
        pepper: Use Pepper syntax instead of the default Chili syntax.
        job_interval: Job scheduler polling interval in milliseconds (0 = disabled).
        memory_limit: Memory limit in MB (0 = unlimited, minimum 1024 MB).
    """

    def __init__(
        self,
        debug: bool = False,
        lazy: bool = False,
        pepper: bool = False,
        job_interval: int = 0,
        memory_limit: float = 0,
    ):
        self.is_tick_loaded = False
        self.is_sub_loaded = False
        self.engine = EngineState(debug, lazy, pepper, job_interval, memory_limit)
        self._hdb_path: Optional[str] = None

    def eval(
        self,
        source: str,
        src_path: Optional[str] = None,
        lazy: bool = False,
    ) -> Any:
        """Evaluate a Chili or Pepper expression string.

        Args:
            source: The expression to evaluate (same syntax as the REPL).
            src_path: Optional logical source path for error messages.
                      Defaults to ``"repl.pep"`` or ``"repl.chi"``
                      depending on the engine's syntax mode.
            lazy: When False (default), DataFrame-shaped results are
                  returned eagerly as ``polars.DataFrame``; when True,
                  results are returned as ``polars.LazyFrame`` for
                  further chained ops + ``.collect()``.

        Returns:
            The result of the evaluation, converted to a Python type.
        """
        if src_path is None:
            src_path = "repl.chi" if self.is_repl_use_chili_syntax() else "repl.pep"
        res = self.engine.eval(source, src_path)
        if lazy and isinstance(res, pl.DataFrame):
            return res.lazy()
        return res

    def get_var(self, id: str) -> Any:
        """Retrieve the value of a variable by name.

        Args:
            id: Variable name.

        Returns:
            The variable's value, converted to a Python type.

        Raises:
            NameError: If the variable does not exist.
        """
        return self.engine.get_var(id)

    def set_var(self, id: str, value: Any):
        """Set or overwrite a variable in the engine.

        Args:
            id: Variable name.
            value: The value to assign (automatically converted from Python).
        """
        return self.engine.set_var(id, value)

    def has_var(self, id: str) -> bool:
        """Check whether a variable exists.

        Args:
            id: Variable name.

        Returns:
            ``True`` if the variable exists, ``False`` otherwise.
        """
        return self.engine.has_var(id)

    def del_var(self, id: str) -> Any:
        """Delete a variable and return its last value.

        Args:
            id: Variable name.

        Returns:
            The deleted variable's value, or ``None`` if it did not exist.
        """
        return self.engine.del_var(id)

    def set_pre_eval_hook(self, name: str) -> None:
        """Register a pre-eval hook on inbound IPC requests.

        ``name(user; handle; query)`` runs before evaluation; its return value
        replaces the query, or a ``raise`` denies the request. Local ``eval`` is
        not gated.

        Args:
            name: Function ``(user; handle; query) -> query'`` already defined.
        """
        return self.engine.set_pre_eval_hook(name)

    def clear_pre_eval_hook(self) -> None:
        """Clear any registered pre-eval hook (requests run unchanged)."""
        return self.engine.clear_pre_eval_hook()

    def get_pre_eval_hook(self) -> "str | None":
        """Return the registered pre-eval hook name, or ``None``."""
        return self.engine.get_pre_eval_hook()

    def set_post_eval_hook(self, name: str) -> None:
        """Register a post-eval audit hook on inbound IPC requests.

        Fired after evaluation with ``name(user; handle; query; result; error)``.
        ``result`` is the evaluated value (Null on error); ``error`` is the error
        string (Null on success). Side effects only — hook errors are logged and
        ignored. Local ``eval`` is not gated.

        Args:
            name: Function ``(user; handle; query; result; error)`` already defined.
        """
        return self.engine.set_post_eval_hook(name)

    def clear_post_eval_hook(self) -> None:
        """Clear any registered post-eval hook (no audit hook fires)."""
        return self.engine.clear_post_eval_hook()

    def get_post_eval_hook(self) -> "str | None":
        """Return the registered post-eval hook name, or ``None``."""
        return self.engine.get_post_eval_hook()

    def set_jobs_deactivate_on_error(self, enabled: bool) -> None:
        """Enable or disable deactivating scheduled jobs after a fire error.

        When enabled, a job that raises on fire is deactivated instead of
        rescheduling every interval. Default disabled.
        """
        return self.engine.set_jobs_deactivate_on_error(enabled)

    def jobs_deactivate_on_error(self) -> bool:
        """Return whether quarantine-on-error is enabled for ``.job``s."""
        return self.engine.jobs_deactivate_on_error()

    def drain(self, id: str) -> Any:
        """Atomically take the accumulated DataFrame and reset the variable.

        Returns all rows accumulated since the last drain (or since
        subscribe) and replaces the variable with a 0-row frame of the
        same schema.  The take and reset happen under a single write-lock,
        so no rows can be lost or duplicated by a concurrent ``upsert``.

        Args:
            id: Variable name (must hold a DataFrame).

        Returns:
            A ``polars.DataFrame`` with the rows that were accumulated.

        Raises:
            NameError: If the variable does not exist.
            RuntimeError: If the variable is not a DataFrame.
        """
        return self.engine.drain(id)

    def upsert(self, id: str, value: Any) -> int:
        """Append rows to an existing DataFrame variable, or create it.

        If the variable does not exist, it is created with the given
        DataFrame.  If it already exists, the rows are appended via
        ``DataFrame.extend``.

        Args:
            id: Variable name.
            value: A ``polars.DataFrame`` (or list) to append.

        Returns:
            The number of rows appended.
        """
        return self.engine.upsert(id, value)

    def insert(self, id: str, value: Any, by: list[str]) -> int:
        """Insert rows into a DataFrame variable, deduplicating by key columns.

        Rows are appended and then deduplicated using a ``group_by(by).last()``
        aggregation, keeping only the latest row per unique key combination.

        Args:
            id: Variable name.
            value: A ``polars.DataFrame`` (or list) to insert.
            by: Column names to deduplicate on.

        Returns:
            The net change in row count (new unique keys added).
        """
        return self.engine.insert(id, value, by)

    def import_source_path(self, relative: str, path: str) -> Any:
        """Import and evaluate a Chili/Pepper source file.

        Args:
            relative: Base path used to resolve relative imports inside
                      the source file.  Pass ``""`` when importing a
                      top-level file.
            path: File system path to the source file.

        Returns:
            The result of evaluating the file.
        """
        return self.engine.import_source_path(relative, path)

    def set_source(self, path: str, src: str) -> Any:
        """Register an in-memory source string under *path*.

        Args:
            path: Logical path associated with the source.
            src: The source code string.

        Returns:
            The index of the registered source entry.
        """
        return self.engine.set_source(path, src)

    def get_source(self, index: int) -> Tuple[str, str]:
        """Retrieve a previously registered source by its index.

        Args:
            index: Zero-based source index.

        Returns:
            A ``(path, source)`` tuple.
        """
        return self.engine.get_source(index)

    def shutdown(self):
        """Shut down the engine and release all IPC handles."""
        self.engine.shutdown()

    def get_displayed_vars(self) -> dict[str, Any]:
        """Return a mapping of variable names to their display strings.

        Functions are shown with their call signatures; other values are
        shown in short-string form.
        """
        return self.engine.get_displayed_vars()

    def list_vars(self, pattern: str) -> list[str]:
        """List engine variables as a Polars DataFrame.

        Args:
            pattern: Prefix filter.  Pass ``""`` to list all variables.

        Returns:
            A ``polars.DataFrame`` with columns ``name``, ``display``,
            ``type``, ``columns``, and ``is_built_in``.
        """
        return self.engine.list_vars(pattern)

    def parse_cache_len(self) -> int:
        """Return the current number of entries in the LRU parse cache."""
        return self.engine.parse_cache_len()

    def get_tick_count(self, index: int = 0) -> int:
        """Return the current tick counter value at *index* (default 0)."""
        return self.engine.get_tick_count(index)

    def tick(self, index: int = 0, inc: int = 1) -> Any:
        """Increment the tick counter at *index* by *inc*.

        Args:
            index: Tick stream index (default 0).
            inc: Amount to add to the counter (default 1).

        Returns:
            The updated tick count.
        """
        return self.engine.tick(index, inc)

    def is_lazy_mode(self) -> bool:
        """Return ``True`` if lazy evaluation mode is enabled."""
        return self.engine.is_lazy_mode()

    def is_repl_use_chili_syntax(self) -> bool:
        """Return ``True`` if the engine uses Chili syntax (not Pepper)."""
        return self.engine.is_repl_use_chili_syntax()

    def fn_call(self, func: str, args: list[Any]) -> Any:
        """Call a registered engine function by name.

        Args:
            func: Function name as registered in the engine.
            args: Positional arguments (converted from Python automatically).

        Returns:
            The function's return value, converted to a Python type.
        """
        return self.engine.fn_call(func, args)

    def write_partitioned_df(
        self,
        df: pl.DataFrame,
        hdb_path: str,
        table: str,
        date: Any,
        sort_columns: Optional[list[str]] = None,
        rechunk: bool = False,
        overwrite: bool = False,
    ) -> int:
        """Write a DataFrame as a date-partitioned Parquet table.

        Args:
            df: The data to write.
            hdb_path: Root directory of the partitioned database.
            table: Table name (sub-directory under each date partition).
            date: Partition date — accepts ``datetime.date`` directly or
                  a ``"YYYY.MM.DD"`` string.
            sort_columns: Optional columns to sort by before writing.
            rechunk: Re-chunk the data into a single contiguous allocation.
            overwrite: If ``True``, overwrite an existing partition.

        Note:
            The Parquet codec is the polars default (**zstd**) and the
            row-group size is auto-sized (clamped when ``sort_columns`` is set,
            else the polars default 262144). Use
            :meth:`write_partitioned_df_custom` for atomic overwrite and a
            per-call compression codec.

        Returns:
            The number of rows written.
        """
        from datetime import date as _date_t
        from datetime import datetime as _dt_t

        if isinstance(date, str):
            partition = _dt_t.strptime(date, "%Y.%m.%d").date()
        elif isinstance(date, (_date_t, _dt_t)):
            partition = date
        else:
            partition = date  # let the engine validate
        sort_cols_arg: Any = pl.Series(
            "sort_columns", sort_columns or [], dtype=pl.Categorical
        )
        return self.fn_call(
            "wpar",
            [
                hdb_path,
                partition,
                table,
                df,
                sort_cols_arg,
                rechunk,
                overwrite,
            ],
        )

    def write_partitioned_df_custom(
        self,
        df: pl.DataFrame,
        hdb_path: str,
        table: str,
        date: Any,
        sort_columns: Optional[list[str]] = None,
        rechunk: bool = False,
        overwrite: bool = False,
        atomic: bool = False,
        compression: Optional[str] = None,
    ) -> int:
        """Write a date-partitioned Parquet table with customized write options.

        Same as :meth:`write_partitioned_df`, plus:

        Args:
            atomic: On overwrite, write the new single shard to a temp file
                and rename it into place so readers never see an empty
                partition directory.
            compression: Optional Parquet codec name (e.g. ``"zstd"``,
                ``"snappy"``); ``None`` keeps the default (**zstd**).

        Returns:
            The number of rows written.
        """
        from datetime import date as _date_t
        from datetime import datetime as _dt_t

        if isinstance(date, str):
            partition = _dt_t.strptime(date, "%Y.%m.%d").date()
        elif isinstance(date, (_date_t, _dt_t)):
            partition = date
        else:
            partition = date  # let the engine validate
        sort_cols_arg: Any = pl.Series(
            "sort_columns", sort_columns or [], dtype=pl.Categorical
        )
        return self.fn_call(
            "wparc",
            [
                hdb_path,
                partition,
                table,
                df,
                sort_cols_arg,
                rechunk,
                overwrite,
                atomic,
                compression,
            ],
        )

    def load_partitioned_df(self, hdb_path: str) -> None:
        """Load a partitioned database from disk.

        After loading, partitions can be queried via the engine.

        Args:
            hdb_path: Root directory of the partitioned database.
        """
        self.fn_call("load", [hdb_path])
        self._hdb_path = hdb_path

    def clear_partitioned_df(self) -> None:
        """Remove all loaded partitioned DataFrames from memory."""
        return self.engine.clear_par_df()

    def table_count(self) -> int:
        """Return the number of partitioned tables currently loaded."""
        return self.engine.table_count()

    def query_plan(self, query: str, hdb_path: Optional[str] = None) -> str:
        """Return the polars query plan for *query* without executing it.

        Internally spins up a temporary **pepper-syntax** lazy-mode engine,
        loads the HDB, evaluates *query* to obtain a ``LazyFrame``, and
        returns its ``describe_plan()`` string. The current engine state
        is unaffected. **Pepper syntax only** — chili-syntax queries will
        fail to parse here even if the calling engine is in chili mode;
        this matches parked-claude's behavior.

        Args:
            query: A pepper-syntax query string (e.g.,
                ``"select last close by sym from ohlcv_1d where date=..."``).
            hdb_path: HDB root directory. Defaults to the most recently
                loaded path on this engine (via ``load_partitioned_df``).
        """
        path = hdb_path if hdb_path is not None else self._hdb_path
        if path is None:
            raise RuntimeError(
                "No HDB path provided and no prior load_partitioned_df() call. "
                "Pass hdb_path= explicitly or call load_partitioned_df() first."
            )
        return self.engine.query_plan(query, path)

    def start_tcp_listener(
        self,
        port: int,
        remote: bool = False,
        users: Optional[list[str]] = None,
    ) -> None:
        """Start a TCP listener on *port* in a background thread.

        Binds synchronously and raises :class:`ChiliError` if the port is
        unavailable. The accept loop runs in the background after a successful
        bind.

        Args:
            port: TCP port number to listen on.
            remote: If ``True``, bind to ``0.0.0.0``; otherwise ``127.0.0.1``.
            users: Allowed usernames; empty list allows any user.

        Raises:
            ChiliError: If bind fails.
        """
        self.engine.start_tcp_listener(port, remote, users or [])

    def set_subscriber_queue_max(self, n: int) -> None:
        """Shed a subscriber whose outbound write queue exceeds ``n`` frames.

        When ``n > 0``, each Publishing subscriber gets a bounded channel and
        writer thread; publish uses non-blocking ``try_send``. A full queue
        disconnects the subscriber. ``0`` (default) disables queue shedding.
        """
        self.engine.set_subscriber_queue_max(n)

    def list_handle(self) -> pl.DataFrame:
        """Return a DataFrame listing all active handles."""
        return self.engine.list_handle()

    def stats(self) -> dict[str, Any]:
        """Return engine statistics as a dictionary.

        Includes lazy mode status, REPL language, partitioned DataFrame
        count, parse cache size, and partition paths.
        """
        return self.engine.stats()

    def load_tick(self) -> None:
        """Load the built-in tick plant source (``src/tick.pep``).

        Evaluates the bundled Pepper script that defines ``.tick.*``
        functions (``createLog``, ``upd``, ``subscribe``, ``unsubscribe``,
        ``eod``).
        """
        if not self.is_tick_loaded:
            tick_path = Path(__file__).parent / "src" / "tick.pep"
            source = tick_path.read_text()
            self.engine.eval(source, "tick.pep")
            self.is_tick_loaded = True

    # Tick functions
    # Feed handler should call .tick.upd
    # Subscriber should call .tick.subscribe and .tick.unsubscribe on tick process
    def init_tick(
        self, schema: Dict[str, pl.DataFrame], log_dir: str, filename: date | str
    ) -> None:
        self.load_tick()
        self.set_var(".tick.schema", schema)
        self.fn_call(".tick.createLog", [log_dir, filename])

    def roll_tick_log(self, log_dir: str, filename: date | str) -> None:
        self.fn_call(".tick.rollLog", [log_dir, filename])

    def publish(self, table: str, data: Any) -> None:
        self.fn_call(".tick.upd", [table, data])

    def eod(self, date: date) -> None:
        self.fn_call(".tick.eod", [date])

    def add_at_time(
        self,
        fn_name: str,
        start_time: datetime,
        description: str = "",
    ) -> int:
        """Schedule a pepper function to fire once at ``start_time``.

        Thin wrapper over chili's ``.job.addAtTime`` registered builtin.
        Backs mdata's PRD §3.2 Option A EOD timer path — replaces their
        Python asyncio timer with a chili-scheduler-owned timer.

        Parameters
        ----------
        fn_name : str
            Name of a **nullary** pepper function in the engine's global
            namespace (e.g., ``my_handler: {[] ...}``). The scheduler
            invokes it as ``fn_name[]`` — passing args via the job spec
            is not supported; use engine variables (``today[]``, ``now[]``,
            or a pre-set global) inside the handler for time context.
        start_time : datetime.datetime
            When to fire. Must be timezone-aware; naive datetimes raise
            ``TypeError``. Attach ``timezone.utc`` explicitly for UTC.
        description : str, optional
            Free-text label, returned by the job-list helpers.

        Returns
        -------
        int
            Job ID. Pass to :meth:`cancel_job` to revoke.

        Notes
        -----
        The chili job scheduler must be running for the timer to fire.
        Construct :class:`ChiliEngine` with ``job_interval > 0`` (in
        milliseconds) to start the scheduler thread.
        """
        return self.engine.add_at_time(fn_name, start_time, description)

    # Subscriber functions
    def load_sub(self) -> None:
        if not self.is_sub_loaded:
            sub_path = Path(__file__).parent / "src" / "sub.pep"
            source = sub_path.read_text()
            self.engine.eval(source, "sub.pep")
            self.is_sub_loaded = True

    # The socket should start with chili://hostname:port
    def subscribe(
        self,
        tick_socket: str,
        topics: Optional[list[str]] = None,
        filters: Optional[dict[str, tuple[str, list[str]]]] = None,
    ) -> None:
        """Subscribe to one or more topics on a tickerplant.

        Args:
            tick_socket: ``chili://host:port`` of the tickerplant.
            topics: list of topic (table) names. Topics that also appear in
                ``filters`` are subscribed with a row-filter instead.
            filters: optional per-topic row filter
                ``{topic: (column, [values])}``. Live broadcasts send only
                matching rows; replay stays unfiltered. Each filtered topic
                uses its own connection.
        """
        self.load_sub()
        filters = filters or {}
        unfiltered = [t for t in (topics or []) if t not in filters]
        if unfiltered or not filters:
            self.fn_call(".sub.init", [tick_socket, unfiltered])
        for topic, (column, values) in filters.items():
            self.fn_call(
                ".sub.initFiltered", [tick_socket, topic, column, list(values)]
            )

    def open_handle(self, socket: str) -> int:
        return self.fn_call(".handle.open", [socket])

    def fsync_handle(self, handle_num: int) -> None:
        """Flush a file handle's buffered data to disk (``fdatasync``).

        Forces all pending writes on the given handle to be persisted,
        ensuring data durability without closing the handle.

        Args:
            handle_num: Handle number to sync.
        """
        self.fn_call(".handle.fsync", [handle_num])

    def sync(self, handle_num: int, query: str) -> Any:
        self.fn_call("set", ["pyHandle", handle_num])
        return self.fn_call("pyHandle", [query])

    def async_(self, handle_num: int, query: str) -> Any:
        neg_handle_num = handle_num if handle_num < 0 else -handle_num
        self.fn_call("set", ["pyHandle", neg_handle_num])
        return self.fn_call("pyHandle", [query])
