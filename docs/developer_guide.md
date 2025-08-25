# Developer Guide

## Setting up

```bash
# Install dev dependencies
poetry install --with dev

# Pre-commit checks
pre-commit install
```

## Code Style

Black + isort. Run `make format`.

## Tests

```
poetry run pytest -q
```

## Release

1. Update `__version__` and `pyproject.toml`.
2. `git tag vX.Y.Z && git push --tags`.
3. `poetry publish --build`. 