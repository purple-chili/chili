# Changelog

All notable changes to this project will be documented in this file.

## [0.7.2] - 2026-02-24

### Removed

- Removed the `%` operator
- Vim syntax highlighting support, chiz will support it soon

### Fixed

- Fixed the handling of Windows paths
- Fixed the symbol token for Windows paths
- Fixed `upsert` and `insert` to support DataFrame as the first argument

## [0.7.1] - 2026-02-22

### Added

- Supported Windows

## [0.7.0] - 2026-02-21

### Added

- Refactored to use chumsky as the parser
- Unified the binaries into one chili binary
- Added `-P` or `--pepper` flag to enable REPL using pepper syntax
- Supported macOS

## [0.6.4] - 2026-02-08

### Added

- Improved the handling of MixedList in the deserialization process to return an empty list when appropriate.
- Modified ListExp, SelectExp, ByExp, Table, Matrix, and Dict to support optional trailing commas.
- Enhanced BracketExp and ColNames to maintain consistency with the new syntax rules.

## [0.6.3] - 2026-01-25

### Added

- Lazy evaluation mode for the runtime
- New built-in function `collect` to collect a lazy DataFrame
- `os.pid` to retrieve the process ID
- `os.version` to retrieve the operating system version
- `os.syntax` to retrieve the syntax type (chili or pepper)

## [0.6.2] - 2025-12-14

### Added

- New built-in function `insert` to insert data from DataFrame or list and keep the last record for each group
- New built-in function `.os.mem` to retrieve memory statistics
- Memory limit command line option for the runtime

### Changed

- Enhanced upsert function to enforce DataFrame type for the first argument
- Updated parser to support new syntax for column definitions

## [0.6.1] - 2025-12-10

### Added

- Support for new syntax highlighting and validation in Chili language
- Short-circuit evaluation for logical operators (`||`, `&&`, `??`) in AST and evaluation logic
- Support for `if` statements with optional `else` blocks for `chili` language
- New binary operators and control keywords in grammar definitions

### Changed

- Enhanced startup banner with vintage feature support and updated graphics
- Updated job function name formatting in EngineState evaluation
- Updated parser to support new control flow structures
- Adjusted tests to reflect changes in parsing and evaluation structure

### Fixed

- Corrected comparison operator in 'lt' function

## [0.6.0] - 2025-12-07

### Added

- Initial release of Chili
- Support for two language syntaxes:
  - chili: a modern programming language similar to JavaScript
  - pepper: a vintage programming language similar to q
- Integration with Polars for data manipulation
- Support for Arrow/Parquet data storage
- Vim syntax highlighting support
