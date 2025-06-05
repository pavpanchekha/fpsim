# FPSim

FPSim simulates and measures floating-point assembly routines. It provides a simple model of various CPU cores and assembles small programs to evaluate their latency or throughput.

Dependencies are managed with [uv](https://github.com/astral-sh/uv). Use `uv sync` to create the virtual environment from `pyproject.toml` and `uv.lock` and run the tool with:

```
uv run python main.py [options] CODE...
```

Apple M1 hardware is currently the best tested and most fully supported platform.

