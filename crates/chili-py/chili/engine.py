"""Python bindings for Chili's ``EngineState`` (Rust ``chili-core``)."""

from typing import Optional

import polars as pl

from .engine_state import EngineState


class ChiliEngine:
    def __init__(self, debug: bool = False, lazy: bool = False, pepper: bool = False):
        self.engine = EngineState(debug, lazy, pepper)

    def eval(self, source: str) -> any:
        return self.engine.eval(source)

    def get_var(self, id: str) -> any:
        return self.engine.get_var(id)

    def set_var(self, id: str, value: any):
        return self.engine.set_var(id, value)

    def has_var(self, id: str) -> bool:
        return self.engine.has_var(id)

    def del_var(self, id: str) -> any:
        return self.engine.del_var(id)

    def import_source_path(self, relative: str, path: str) -> any:
        return self.engine.import_source_path(relative, path)

    def set_source(self, path: str, src: str) -> any:
        return self.engine.set_source(path, src)

    def get_source(self, index: int) -> tuple[str, str]:
        return self.engine.get_source(index)

    def shutdown(self):
        self.engine.shutdown()

    def get_displayed_vars(self) -> dict[str, any]:
        return self.engine.get_displayed_vars()

    def list_vars(self, pattern: str) -> list[str]:
        return self.engine.list_vars(pattern)

    def parse_cache_len(self) -> int:
        return self.engine.parse_cache_len()

    def get_tick_count(self) -> int:
        return self.engine.get_tick_count()

    def tick(self, inc: int) -> any:
        return self.engine.tick(inc)

    def is_lazy_mode(self) -> bool:
        return self.engine.is_lazy_mode()

    def is_repl_use_chili_syntax(self) -> bool:
        return self.engine.is_repl_use_chili_syntax()

    def fn_call(self, func: str, args: list[any]) -> any:
        return self.engine.fn_call(func, args)

    def write_partitioned_df(
        self,
        df: pl.DataFrame,
        hdb_path: str,
        table: str,
        date: str,
        sort_columns: Optional[list[str]] = None,
        rechunk: bool = False,
    ) -> int:
        return self.fn_call("wpar", [df, hdb_path, table, date, sort_columns, rechunk])

    def load_partitioned_df(self, hdb_path: str) -> None:
        return self.fn_call("load", [hdb_path])

    def clear_partitioned_df(self) -> None:
        return self.engine.clear_par_df()
