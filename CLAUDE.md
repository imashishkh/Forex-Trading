# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `pytest`
- Run a single test: `pytest test_file.py::test_function -v`
- Run tests with coverage: `pytest --cov=.`
- Check imports: `python test_imports.py`

## Code Style Guidelines
- **Imports**: Group imports: standard library, third-party libraries, local modules. Import specific functions when possible.
- **Formatting**: Follow PEP 8 conventions. Use 4 spaces for indentation.
- **Types**: Use type hints for function parameters and return values. Import types from `typing`.
- **Docstrings**: Use triple-quoted docstrings. Include Args, Returns, and Raises sections.
- **Naming**: Use snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants.
- **Error Handling**: Use specific exception classes. Handle exceptions with try/except blocks with detailed error messages.
- **Structure**: Organize agent classes with initialize, process_message, run_task and shutdown methods.
- **Logging**: Use the logger from utils.logger or logging module with appropriate log levels.