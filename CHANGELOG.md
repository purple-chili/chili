# Changelog

All notable changes to this project will be documented in this file.

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
