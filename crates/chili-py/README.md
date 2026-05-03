# Chili Sauce 🌶️

Python bindings for [Chili](https://purple-chili.github.io/)'s `EngineState`, powered by [PyO3](https://pyo3.rs/).

## Installation

```bash
pip install chili-sauce
```

Requires Python ≥ 3.10.

## Quick Start

```python
from chili import ChiliEngine

engine = ChiliEngine()

# Evaluate expressions
engine.eval("1 + 2")  # => 3

# Variable management
engine.set_var("x", 42)
engine.get_var("x")  # => 42

# Work with Polars DataFrames
import polars as pl

df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
engine.set_var("df", df)
engine.get_var("df")
```

## Features

- **Evaluate** Chili or Pepper expressions from Python
- **Variable management** — get, set, delete, and list variables
- **Polars integration** — pass DataFrames bidirectionally between Python and Chili
- **Partitioned storage** — write and load date-partitioned Parquet tables
- **IPC / TCP** — start a TCP listener for remote connections
- **Tick plant** — built-in pub/sub infrastructure for real-time data

## Type Mapping

| Python type        | Chili type        |
|--------------------|-------------------|
| `int`              | `Int`             |
| `float`            | `Float`           |
| `bool`             | `Bool`            |
| `str`              | `Symbol`          |
| `bytes`            | `String`          |
| `None`             | `Null`            |
| `list`             | `MixedList`       |
| `dict`             | `Dict`            |
| `datetime.date`    | `Date`            |
| `datetime.time`    | `Time`            |
| `datetime.datetime`| `Datetime`        |
| `polars.DataFrame` | `DataFrame`       |

## Development

```bash
# Build and install in development mode
maturin develop --release --manifest-path crates/chili-py/Cargo.toml

# Run tests
pytest crates/chili-py/tests/
```
