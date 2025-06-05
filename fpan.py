
FPAN333 = [
    (1, 2, "ts"), (3, 4, "ts"), (5, 6, "ts"),
    (1, 3, "fts"), (2, 6, "add"), (4, 5, "ts"),
    (1, 4, "fts"), (2, 3, "ts"), (3, 5, "add"),
    (3, 4, "ts"), (2, 3, "ts"),
    (1, 2, "fts"), (3, 4, "add"),
    (2, 3, "fts")
]

def interpret(fpan):
    def impl(code, *args, **kwargs):
        args = [None] + list(args)
        for a, b, t in fpan:
            out = kwargs[t](code, args[a], args[b])
            if isinstance(out, tuple):
                args[a], args[b] = out
            else:
                args[a] = out
                args[b] = None
        return [x for x in args if x is not None]
    return impl


def critical_gates(fpan):
    """Return the list of gate indices on the critical path of ``fpan``.

    Each entry of ``fpan`` is a tuple ``(a, b, op)``.  The gate takes two
    inputs and writes two outputs back to registers ``a`` and ``b``.  The
    latency to each output depends on ``op``:

    ``ts``  -> ``(3, 15)``
    ``fts`` -> ``(3, 9)``
    ``add`` -> ``(3, 3)``
    ``cts`` -> ``(3, 11)``

    The critical path is the dependency chain with the largest total latency.
    Gate indices are returned 0-based.
    """

    if not fpan:
        return []

    n_regs = max(max(a, b) for a, b, _ in fpan)

    reg_time = [0] * (n_regs + 1)
    reg_src = [None] * (n_regs + 1)

    gate_info = []

    latencies = {
        "ts": (3, 15),
        "fts": (3, 9),
        "add": (3, 3),
        "cts": (3, 11),
    }

    for i, (ra, rb, op) in enumerate(fpan):
        ia_time = reg_time[ra]
        ib_time = reg_time[rb]
        ia_src = reg_src[ra]
        ib_src = reg_src[rb]
        start = max(ia_time, ib_time)
        gate_info.append((ia_src, ia_time, ib_src, ib_time, start))

        a_lat, b_lat = latencies[op]
        reg_time[ra] = start + a_lat
        reg_src[ra] = i
        reg_time[rb] = start + b_lat
        reg_src[rb] = i

    max_time = max(reg_time[1:])
    worklist = [reg_src[r] for r in range(1, n_regs + 1)
                if reg_time[r] == max_time and reg_src[r] is not None]

    crit = set()
    while worklist:
        idx = worklist.pop()
        if idx is None or idx in crit:
            continue
        crit.add(idx)
        ia_src, ia_time, ib_src, ib_time, start = gate_info[idx]
        if ia_time == start and ia_src is not None:
            worklist.append(ia_src)
        if ib_time == start and ib_src is not None:
            worklist.append(ib_src)

    return set(crit)
        
def criticalize(fpan):
    crit = critical_gates(fpan)
    out = []
    for i, (a, b, t) in enumerate(fpan):
        if t == "ts":
            out.append((a, b, "cts" if i in crit else "ts"))
        else:
            out.append((a, b, t))
    if out != fpan:
        return criticalize(out)
    else:
        return out
