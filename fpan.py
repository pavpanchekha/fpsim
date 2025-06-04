
FPAN333 = [
    (1, 2, "cts"), (3, 4, "ts"), (5, 6, "cts"),
    (1, 3, "fts"), (2, 6, "add"), (4, 5, "ts"),
    (1, 4, "fts"), (2, 3, "cts"), (3, 5, "add"),
    (3, 4, "ts"), (2, 3, "cts"),
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
        
        


