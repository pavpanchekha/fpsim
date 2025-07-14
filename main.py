from functools import wraps
import inspect
import sys
import fpan
import math

from assembler import ARM, X86, ISA, Sig, Assembler
from cores import Core, CPU, CORES
from measure import compile_run

CODES = {}


# Written by ChatGPT 4o
def num_args(fn):
    sig = inspect.signature(fn)
    return len([
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ])

class Listing:
    def __init__(self, f, kwargs, argnum=None):
        self.f = f
        self.argnum = argnum if argnum is not None else num_args(f) - 1
        self.kwargs = kwargs

    def __call__(self, code, *args):
        return self.f(code, *args, **self.kwargs)

    def set_kwargs(self, kwargs):
        return Listing(self.f, kwargs)

    def replicate(self, instances=1):
        def f(code, *iargs, **kwargs):
            assert len(iargs) == instances * self.argnum, "Wrong number of arguments to replicated inputs"
            oargs = []
            iargs = list(range(-instances*self.argnum, 0))
            for instance in range(instances):
                this_iargs = iargs[instance*self.argnum : (instance+1)*self.argnum]
                this_oargs = self.f(code, *this_iargs, **self.kwargs)
                oargs.append(max(*this_oargs))
            copies = int(math.ceil(len(iargs) / max(1, len(oargs))))
            latency_ties = list(zip(iargs, [oarg for oarg in oargs for i in range(copies)]))
            for iarg, oarg in latency_ties:
                code.code.append((iarg, code.isa.mov, [oarg]))
            return tuple(oargs)
        return Listing(f, self.kwargs, argnum=self.argnum*instances)

## Start extant algorithms

def algorithm(_func=None, **kwargs):
    def decorator(f):
        global CODES
        name = f.__name__
        CODES[name] = Listing(f, kwargs)

        @wraps(f)
        def wrapper(code, *args):
            return f(code, *args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)

@algorithm
def fts(code, a, b):
    if code.isa == ARM:
        s = code.fadd(a, b)
        return s, code.fsub(b, code.fsub(s, a))
    elif code.isa == X86:
        s = code.vaddsd(a, b)
        return s, code.vsubsd(b, code.vsubsd(s, a))

@algorithm
def ts(code, a, b):
    if code.isa == ARM:
        s = code.fadd(a, b)
        bb = code.fsub(s, a)
        return s, code.fadd(code.fsub(a, code.fsub(s, bb)), code.fsub(b, bb))
    elif code.isa == X86:
        s = code.vaddsd(a, b)
        bb = code.vsubsd(s, a)
        return s, code.vaddsd(code.vsubsd(a, code.vsubsd(s, bb)), code.vsubsd(b, bb))

def sort2(code, a, b):
    if code.isa == ARM:
        aa, ab = code.fabs(a), code.fabs(b)
        cond = code.fcmp(aa, ab)
        x = code.fcsel(cond, b, a)
        y = code.fcsel(cond, a, b)
    elif code.isa == X86:
        aa, ab = code.vandpd(a, "ABS_MASK"), code.vandpd(b, "ABS_MASK")
        cond = code.vcmpltsd(aa, ab)
        x = code.vblendvpd(a, b, cond)
        y = code.vblendvpd(b, a, cond)
    return x, y

def add(code, a, b):
    if code.isa == ARM:
        return code.fadd(a, b)
    elif code.isa == X86:
        return code.vaddsd(a, b)


def sub(code, a, b):
    if code.isa == ARM:
        return code.fsub(a, b)
    elif code.isa == X86:
        return code.vsubsd(a, b)


def fabs(code, a):
    if code.isa == ARM:
        return code.fabs(a)
    elif code.isa == X86:
        return code.vandpd(a, "ABS_MASK")


@algorithm
def at0(code, a, b):
    a, b = sort2(code, a, b)
    return fts(code, a, b)


@algorithm
def at1(code, a, b):
    s = add(code, a, b)
    a, b = sort2(code, a, b)
    return s, sub(code, b, sub(code, s, a))


@algorithm
def at2(code, a, b):
    s = add(code, a, b)
    aa = sub(code, s, b)
    bb = sub(code, s, a)
    if code.isa == ARM:
        cond = code.fcmp(code.fabs(b), code.fabs(a))
        x = code.fcsel(cond, b, a)
        xx = code.fcsel(cond, bb, aa)
    elif code.isa == X86:
        aa, ab = fabs(code, a), fabs(code, b)
        cond = code.vcmpltsd(aa, ab)
        x = code.vblendvpd(a, b, cond)
        xx = code.vblendvpd(aa, bb, cond)
    return s, sub(code, x, xx)


@algorithm
def at3(code, a, b):
    s = add(code, a, b)
    ifa = sub(code, b, sub(code, s, a))
    ifb = sub(code, a, sub(code, s, b))
    aa, bb = fabs(code, a), fabs(code, b)
    if code.isa == ARM:
        cond = code.fcmp(aa, bb)
        return s, code.fcsel(cond, ifb, ifa)
    elif code.isa == X86:
        cond = code.vcmpltsd(aa, bb)
        return s, code.vblendvpd(ifa, ifb, cond)

@algorithm
def at4(code, a, b):
    s = add(code, a, b)
    ifa = sub(code, b, sub(code, s, a))
    ifb = sub(code, a, sub(code, s, b))
    aa, bb = fabs(code, a), fabs(code, b)
    if code.isa == ARM:
        cond = code.fcmge(aa, bb)
        return s, code.bsl(cond, ifb, ifa)
    elif code.isa == X86:
        cond = code.vcmpltsd(aa, bb)
        return s, code.vblendvpd(ifa, ifb, cond)

@algorithm(ts=ts)
def ddadd(code, x0, y0, x1, y1, *, ts=ts):
    x0, y0 = ts(code, x0, y0)
    x1, y1 = ts(code, x1, y1)
    y0     = add(code, y0, x1)
    x0, y0 = ts(code, x0, y0)
    y0     = add(code, y0, y1)
    x0, y0 = ts(code, x0, y0)
    return x0, y0

@algorithm(ts=ts, ts2=fts)
def madd(code, x0, y0, x1, y1, *, ts=ts, ts2=fts):
    x0, y0 = ts(code, x0, y0)
    x1, y1 = ts(code, x1, y1)
    x0, x1 = ts(code, x0, x1)
    y0     = add(code, y0, y1)
    y0     = add(code, y0, x1)
    x0, y0 = ts2(code, x0, y0)
    return x0, y0

@algorithm(ts=ts, cts=ts, fts=fts, add=add)
def fpan333(code, a, b, c, d, e, f, *, ts=ts, cts=ts, fts=fts, add=add):
    return fpan.interpret(fpan.FPAN333)(code, a, b, c, d, e, f, ts=ts, cts=cts, fts=fts, add=add)

@algorithm(ts=ts, cts=ts, fts=fts, add=add)
def fpan444(code, a, b, c, d, e, f, g, h, *, ts=ts, cts=ts, fts=fts, add=add):
    return fpan.interpret(fpan.criticalize(fpan.FPAN444_3))(
        code, a, b, c, d, e, f, g, h,
        ts=ts, cts=cts, fts=fts, add=add)

VERBOSE = False

def get_code(name):
    if "[" in name:
        prefix, rest = name.split("[", 1)
        args, suffix = rest.split("]", 1)
        kwargs = {}
        for arg in args.split(","):
            key, value = arg.split("=")
            if value.isdigit():
                value = int(value)
            else:
                value = get_code(value)
            kwargs[key] = value
        return get_code(prefix + suffix).set_kwargs(kwargs)
    else:
        return CODES[name]

def main():
    import argparse
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Simulate CPU performance for TwoSum variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available codes: {}".format(", ".join(sorted(CODES.keys())))
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--core', type=str, default="P", choices=CORES.keys(), help='P-core or E-core')
    parser.add_argument('--instances', type=int, default=1, help='Number of instances')
    parser.add_argument('--mode', choices=['simulate', 'measure', 'both'], default='both',
                        help='Run simulation, measurement, or both')
    parser.add_argument('--ssh', type=str, default=None,
                        help='Compile and run on the specified SSH host')
    parser.add_argument('codes', nargs=argparse.REMAINDER, help='Functions to simulate')
    args = parser.parse_args()

    # Assign flags
    global VERBOSE
    VERBOSE = args.verbose
    num_instances = args.instances
    codes = args.codes if args.codes else CODES.keys()

    # Simulate and/or measure and print results
    for name in codes:
        core = CORES[args.core]
        code = get_code(name).replicate(num_instances)
        results = []
        if args.mode != "measure":
            sim = CPU(core, code, verbose=args.verbose).simulate()
            results.append(f"{sim:>5.2f} simulated latency")
        if args.mode != "simulate":
            meas = compile_run(code, core, ssh_host=args.ssh)
            results.append(f"{meas:>5.2f} measured latency")
        instructions = len(Assembler(core.isa).exec(code.f, *range(-code.argnum, 0)).code)
        results.append(f"{instructions:>2} instructions")
        print(f"{name:>20}: {', '.join(results)}")

if __name__ == "__main__":
    main()
