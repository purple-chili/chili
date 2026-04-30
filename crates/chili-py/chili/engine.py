"""Python bindings for Chili's ``EngineState`` (Rust ``chili-core``)."""

from typing import Any, Optional, Tuple

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
    """

    def __init__(self, debug: bool = False, lazy: bool = False, pepper: bool = False):
        self.engine = EngineState(debug, lazy, pepper)

    def eval(self, source: str) -> Any:
        """Evaluate a Chili or Pepper expression string.

        Args:
            source: The expression to evaluate (same syntax as the REPL).

        Returns:
            The result of the evaluation, converted to a Python type.
        """
        return self.engine.eval(source)

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

    def get_tick_count(self) -> int:
        """Return the current tick counter value."""
        return self.engine.get_tick_count()

    def tick(self, inc: int) -> Any:
        """Increment the tick counter.

        Args:
            inc: Amount to add to the counter.

        Returns:
            The updated tick count.
        """
        return self.engine.tick(inc)

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
        date: str,
        sort_columns: Optional[list[str]] = None,
        rechunk: bool = False,
        overwrite: bool = False,
    ) -> int:
        """Write a DataFrame as a date-partitioned Parquet table.

        Args:
            df: The data to write.
            hdb_path: Root directory of the partitioned database.
            table: Table name (sub-directory under each date partition).
            date: Partition date string (e.g. ``"2024.01.01"``).
            sort_columns: Optional columns to sort by before writing.
            rechunk: Re-chunk the data into a single contiguous allocation.
            overwrite: If ``True``, overwrite an existing partition.

        Returns:
            The number of rows written.
        """
        return self.fn_call(
            "wpar", [df, hdb_path, table, date, sort_columns, rechunk, overwrite]
        )

    def load_partitioned_df(self, hdb_path: str) -> None:
        """Load a partitioned database from disk.

        After loading, partitions can be queried via the engine.

        Args:
            hdb_path: Root directory of the partitioned database.
        """
        self.fn_call("load", [hdb_path])

    def clear_partitioned_df(self) -> None:
        """Remove all loaded partitioned DataFrames from memory."""
        return self.engine.clear_par_df()

    def start_tcp_listener(
        self,
        port: int,
        remote: bool = False,
        users: Optional[list[str]] = None,
    ) -> None:
        """Start a TCP listener on *port* in a background thread.

        The listener accepts incoming IPC connections (Q or Chili
        protocol), performs authentication, and dispatches each
        connection to its own handler thread.

        Args:
            port: TCP port number to listen on.
            remote: If ``True``, bind to ``0.0.0.0`` (accept remote connections).
                    Otherwise bind to ``127.0.0.1`` (localhost only).
            users: Optional list of usernames allowed to authenticate.
                   An empty list (the default) allows any user.
        """
        self.engine.start_tcp_listener(port, remote, users or [])

    def stats(self) -> dict[str, Any]:
        """Return engine statistics as a dictionary.

        Includes lazy mode status, REPL language, partitioned DataFrame
        count, parse cache size, and partition paths.
        """
        return self.engine.stats()
