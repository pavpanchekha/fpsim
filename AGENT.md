# Agent Instructions

Dependencies are managed with `uv`. Use `uv run` to run the tool, like so:

```
$ uv run python3 main.py ts
     ts: 15.02 simulated latency, 15.42 measured latency
```

You can add `--instances 3` to load the simulated CPU more heavily; it
works by running the same assembly snippet several times in parallel
in the same loop. You can pass `--mode measure` or `--mode simulate`
to just do simulation or measurement; you often want to skip
measurement because it involves running assembly code. If you *do*
want to measure, use `--core CL` to switch to Coffee Lake mode, which
is at least X86 and probably will work in the VM.
