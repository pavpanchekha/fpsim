# FPSim

FPSim simulates and measures floating-point assembly routines with a
focus on floating-point accumulation networks (FPANs). It assembles
and optimizes small programs to evaluate their latency or throughput,
either on the user's machine or in a simple CPU simulator.

## Getting Started

Dependencies are managed with [uv](https://github.com/astral-sh/uv).
Use `uv sync` to create a virtual environment and run the tool with:

```
$ uv run python3 main.py ts
     ts: 15.02 simulated latency, 15.42 measured latency
```

This runs a simple TwoSum routine both in simulation and on your
machine, and then prints its latency.

The Apple M1 is currently the best tested and most fully supported
platform. Support for x86 is planned but currently incomplete, though
you can test it by passing `--core CL` and running on an x86 machine.

## Features

FPSim offers a couple of interesting features, including:

- Optimal register allocation using an ILP solver (via PuLP)
- A fine-grained queue-and-port model of Apple M1 execution
- Both low-latency and high-throughput TwoSum variants
- Analytical tuning tools for FPANs, such as finding critical paths
