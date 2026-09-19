# Set up the environment

## When to use

Before running anything, on a fresh clone or a new machine.

## Workflow

```bash
uv sync
```

That reads `pyproject.toml`, installs the versions pinned in `uv.lock` into `.venv/`, and downloads
CPython 3.12 if absent (pinned in `.python-version`). Then prefix commands with `uv run`, or
activate `.venv`.

```bash
uv run jupyter lab      # notebook deps are in the "notebook" group, installed by default
```

## Notes

- `requires-python` is capped at `<3.13`. Uncapped, uv resolves against 3.14, where rasterio,
  geopandas and torch have no wheels. If you widen it, re-run `uv lock` and actually `uv sync`.
- PyPI's torch gives a CPU/MPS build on macOS and a bundled-CUDA build on Linux. The published model
  was trained against cu118; `pyproject.toml` carries a commented `[[tool.uv.index]]` block for
  pinning that exactly.

## Credentials

```bash
aws sts get-caller-identity     # confirm you are in the project AWS account
```

Earth Engine is only needed to pull *new* imagery: `ee.Authenticate()` once, then `ee.Initialize()`.

## References

- Longer walkthrough: `docs/ONBOARDING.md`, "Getting set up"
