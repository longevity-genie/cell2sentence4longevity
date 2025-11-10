# Agent Guidelines

This document outlines the coding standards and practices to follow when working on this project.

## Error Handling

- **Avoid nested try-catch blocks**: Do not nest try-catch blocks unless absolutely necessary
- **Eliot logging**: Avoid try-catch inside eliot catching unless absolutely necessary, as `action.log` can work with errors
- **Minimize try-catch usage**: Try to avoid excessive try-catch-except blocks, especially inside loggable actions

## Code Quality

- **No placeholders**: Do not put placeholders like `/my/custom/path/` in real code
- **Type hints**: Always use type hints in Python projects
- **No relative imports**: Do not use relative imports! Always use absolute imports

## Testing

- **Integration tests**: When asked to write integration tests, run real requests and work with real data. Do not mock stuff unless explicitly asked or if dealing with multi-gigabyte files

## Dependency Management

- **Use `uv`**: In Python projects, always use `uv` with `project.toml`
- **Avoid `uv pip install`**: Do not use `uv pip install`. Use `uv sync` to resolve dependencies and `uv add` to add new ones
- **Version management**: Never hardcode version into `__init__.py` file (use `project.toml` instead)

## Data Processing

- **Prefer Polars**: Prefer Polars over Pandas for data processing

## Logging

- **Eliot logging**: Use eliot as the default logging library using the `with start_action(...) as action` pattern

## CLI Development

- **Use Typer**: Always use the typer library when making CLI applications

## Data Validation

- **Pydantic version**: When using pydantic, assume pydantic 2 by default. Never use outdated pydantic 1

## Debugging

- **Print statements**: If debugging in bash with Python print statements, try to avoid using "!" character

## UI/UX Libraries

- **Dashly duplicates**: If using dashly, avoid allowing duplicates unless the user wants them explicitly

## Documentation

- **Markdown file placement**: For markdown files you generate, unless it is a README or dataset card, put them in the `docs` folder
- **Avoid excessive markdown**: Avoid creating many markdown files when explaining what you changed in the code unless it is a README or explicitly requested
- **No change summaries**: Do NOT create changes summaries as files unless explicitly asked

