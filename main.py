import typing
from dataclasses import dataclass
from functools import wraps
import inspect
import pulp
import tempfile
from pathlib import Path
import subprocess
import sys
import time
import fpan
import math

@dataclass
class Sig:
    args : list[str]
    suffix : typing.Union[None, str]
    
    def __init__(self, *args, suffix=None):
        self.args = args
        self.suffix = suffix

@dataclass
class ISA:
    registers : int
    prefix : str
    instructions: dict[str, list[str]]

ARM = ISA(
    registers = 32,
    prefix = "d",
    instructions = {
        "fadd": Sig("out", "in", "in"),
        "fsub": Sig("out", "in", "in"),
        "fabs": Sig("out", "in"),
        "fcmp": Sig("out flags", "in", "in"),
        "fcsel": Sig("out", "in flags", "in", "in", suffix="lt")
    }
)

X86 = ISA(
    registers = 16,
    prefix = "xmm",
    instructions = {
        "vaddsd": Sig("out", "in", "in"),
        "vsubsd": Sig("out", "in", "in"),
        "vandpd": Sig("out", "in", "const"),
        "vcmpltsd": Sig("out", "in", "in"),
        "vpcmpgtq": Sig("out", "in", "in"),
        "vblendvpd": Sig("out", "in", "in", "in"),
    },
)

@dataclass
class Core:
    isa : ISA
    latencies : dict[str, tuple[int, list[int]]]
    priority : list[int]

# Apple M1
PCORE = Core(ARM, {
    "fadd": (3, [11, 12, 13, 14]),
    "fsub": (3, [11, 12, 13, 14]),
    "fabs": (2, [11, 12, 13, 14]),
    "fcmp": (2, [11]),
    "fcsel": (2, [13, 14]),
}, priority=[12, 13, 14, 11])

ECORE = Core(ARM, {
    "fadd": (3, [6, 7]),
    "fsub": (3, [6, 7]),
    "fabs": (2, [6, 7]),
    "fcmp": (2, [6]),
    "fcsel": (2, [6, 7]),
}, priority = [7, 6])

# Coffee Lake
CL_CORE = Core(X86, {
    "vaddsd": (4, [0, 1]),
    "vsubsd": (4, [0, 1]),
    "vandpd": (1, [0, 1, 5]),
    "vcmpltsd": (4, [0, 1]),
    "vpcmpgtq": (3, [5]),
    "vblendvpd": (2, [0, 1, 5]),
}, priority=[0, 1, 5])

CORES = {
    "E": ECORE,
    "P": PCORE,
    "CL": CL_CORE,
}

CODES = {}

@dataclass
class Assembler:
    isa: ISA
    ip: int
    code: list[tuple[int, str, list[int]]]

    def __init__(self, isa: ISA):
        self.isa = isa
        self.ip = 0
        self.code = []
        self.flags = set()
        self.curflags = None

    def push_instruction(self, name, args):
        out = self.ip
        self.ip += 1
        signature = self.isa.instructions[name].args
        assert "out" in signature[0].split()
        assert len(args) == len(signature[1:]), f"{name} wrong number of arguments"
        if "flags" in signature[0].split():
            self.flags.add(out)
            self.curflags = out
        for arg, sig in zip(args, signature[1:]):
            if "flags" in sig.split():
                assert arg in self.flags, f"{name} argument not a flags register"
                assert arg == self.curflags, f"{name} argument is not current flags"
            elif "const" in sig.split():
                assert isinstance(arg, str), f"{name} argument not a constant"
            elif "in" in sig.split():
                assert isinstance(arg, int), f"{name} argument not a register"
        self.code.append([out, name, args])
        return out
    
    def exec(self, f, *args, **kwargs):
        f(self, *args, **kwargs)
        return self

    def __getattr__(self, name):
        return lambda *args: self.push_instruction(name, args)

# Written by ChatGPT 4o
def num_args(fn):
    sig = inspect.signature(fn)
    return len([
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ])

ASM_ITERATIONS = 100_000_000

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
            for instance in range(instances):
                this_iargs = iargs[instance*self.argnum : (instance+1)*self.argnum]
                this_oargs = self.f(code, *this_iargs, **self.kwargs)
                oargs.append(max(*this_oargs))
            return tuple(oargs)
        return Listing(f, self.kwargs, argnum=self.argnum*instances)

    def to_asm(self, isa: ISA, name, fd):
        l = Assembler(isa)
        iargs = range(-self.argnum, 0)
        oargs = self.f(l, *iargs, **self.kwargs)

        copies = int(math.ceil(len(iargs) / max(1, len(oargs))))
        latency_ties = list(zip(iargs, [oarg for oarg in oargs for i in range(copies)]))

        regs = set()
        regstart = {}
        regend = {}
        for arg in iargs:
            regstart[arg] = -1 # before start
            regs.add(arg)
        for out, op, args in l.code:
            regstart[out] = out
            regend[out] = out
            regs.add(out)
            for arg in args:
                if isinstance(arg, str): continue
                regend[arg] = out
        for arg in oargs:
            regend[arg] = l.ip # past end
            regs.add(arg)

        if VERBOSE: print("Performing register allocation with PuLP")
        start = time.time()
        model = pulp.LpProblem(sense=pulp.LpMinimize)

        at = {}
        for vreg in regs:
            for preg in range(isa.registers):
                at[(vreg, preg)] = pulp.LpVariable(f"at_{vreg}_{preg}", cat=pulp.LpBinary)

        for vreg in regs:
            if vreg in l.flags: continue # Flags don't take up an ISA register
            # Each v-reg is in exactly one p-reg
            model += sum(at[vreg, preg] for preg in range(isa.registers)) == 1

        for reg1 in regs:
            for reg2 in regs:
                if reg1 == reg2: break
                if regstart[reg1] < regend[reg2] and regstart[reg2] < regend[reg1]:
                    for preg in range(isa.registers):
                        # Can't put two interfering v-regs in the same p-reg
                        model += at[(reg1, preg)] + at[(reg2, preg)] <= 1

        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        assert model.status == pulp.LpStatusOptimal, "Too many registers, could not allocate"

        assignment = {}
        for vreg in regs:
            for preg in range(isa.registers):
                if at[(vreg, preg)].value() == 1:
                    assignment[vreg] = preg
        if VERBOSE: print("  Done in {:.02f}ms with {} registers".format(
                (time.time() - start) * 1000,
                len(set(assignment.values())),
        ))

        print(".text", file=fd)
        print(f".global _{name}, {name}", file=fd)
        print(f"{name}:", file=fd)
        print(f"_{name}:", file=fd)

        # Prologue
        if isa == ARM: # No need on x86, all xmm registers caller-saved
            print("  stp    q8,  q9,  [sp, #-32]!", file=fd)
            print("  stp    q10, q11, [sp, #-32]!", file=fd)
            print("  stp    q12, q13, [sp, #-32]!", file=fd)
            print("  stp    q14, q15, [sp, #-32]!", file=fd)

        if isa == ARM:
            for ireg in iargs:
                print(f"  fmov d{assignment[ireg]}, #0.0", file=fd)
            print(f"  ldr x9, ={ASM_ITERATIONS}", file=fd)
        elif isa == X86:
            for ireg in iargs:
                print(f"  xorpd xmm{assignment[ireg]}, xmm{assignment[ireg]}", file=fd)
            print(f"  mov r11, [rip+LOOP_ITERS]", file=fd)

        print(".p2align 4", file=fd)
        print("1:", file=fd)

        PREFIX = isa.prefix
        for out, op, args in l.code:
            signature = isa.instructions[op]
            arglist = []
            for arg, sig in zip([out] + list(args), signature.args):
                if "flags" in sig.split():
                    continue
                elif isinstance(arg, str):
                    arglist.append(f"[rip+{arg}]")
                else:
                    arglist.append(isa.prefix + str(assignment[arg]))
            if signature.suffix:
                arglist.append(signature.suffix)
            print(f"  {op} {', '.join(arglist)}", file=fd)

        # At the tail we need to move output registers back into input
        # registers to enforce loop-carried dependency.
        if isa == ARM:
            for dst, src in latency_ties:
                if assignment[dst] == assignment[src]: continue
                print(f"  fmov d{assignment[dst]}, d{assignment[src]}", file=fd)
        elif isa == X86:
            for dst, src in latency_ties:
                if assignment[dst] == assignment[src]: continue
                print(f"  movsd xmm{assignment[dst]}, xmm{assignment[src]}", file=fd)

        if isa == ARM:
            print("  subs x9, x9, #1", file=fd)
            print("  b.ne 1b", file=fd)
        elif isa == X86:
            print("  dec r11", file=fd)
            print("  jnz 1b", file=fd)
        if isa == ARM: # No need on x86, all xmm caller-saved
            print("  ldp    q14, q15, [sp], #32", file=fd)
            print("  ldp    q12, q13, [sp], #32", file=fd)
            print("  ldp    q10, q11, [sp], #32", file=fd)
            print("  ldp    q8,  q9,  [sp], #32", file=fd)
        print("  ret", file=fd)

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

class CPU:
    def __init__(self, core: Core, code):
        self.core = core
        l = Assembler(core.isa)
        self.oargs = code(l, *range(-code.argnum, 0))
        self.code = l.code

        self.last = max(pc for pc, opcode, deps in self.code)
        self.status = {}

        self.cycle = 0
        self.completions = 0

    def dispatch(self, instruction, units):
        pc, opcode, deps = instruction
        latency, ports = self.core.latencies[opcode]
        if pc in self.status:
            return None
        if not all(self.status.get(dep) == 0 for dep in deps if isinstance(dep, int) and dep >= 0):
            return None
        for unit in self.core.priority:
            if unit not in units and unit in ports:
                if VERBOSE: print(f"[{self.cycle:>5}] {pc} ({opcode}) start on u{unit}")
                self.status[pc] = latency
                units[unit] = True
                return

    def tick(self):
        units = {}
        for instruction in self.code:
            self.dispatch(instruction, units)
        for pc, latency in self.status.items():
            if latency == 0: continue
            self.status[pc] = latency - 1
        if all(self.status.get(oarg) == 0 for oarg in self.oargs):
            if VERBOSE: print(f"Instance completed on cycle {self.cycle}")
            self.completions += 1
            self.status = {}
        self.cycle += 1

    def simulate(self, cycles=10000):
        for i in range(cycles):
            self.tick()
        return self.cycle / self.completions

# Written by ChatGPT o4-mini
DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

extern void null_loop(void);
extern void bench_loop(void);

int main(void) {
#ifdef __APPLE__
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);

    uint64_t start = mach_absolute_time();
#else
    uint64_t start = __builtin_ia32_rdtsc();
#endif
    null_loop();
#ifdef __APPLE__
    uint64_t end = mach_absolute_time();
#else
    uint64_t end = __builtin_ia32_rdtsc();
#endif
    uint64_t baseline = end - start;

#ifdef __APPLE__
    start = mach_absolute_time();
#else
    start = __builtin_ia32_rdtsc();
#endif
    bench_loop();
#ifdef __APPLE__
    end = mach_absolute_time();
#else
    end = __builtin_ia32_rdtsc();
#endif

    double ticks = end - start - baseline;
#ifdef __APPLE__
    double cycles_per = ticks * timebase.numer / timebase.denom * 3.2;
#else
    double cycles_per = ticks;
#endif
    printf("%.2f cycles elapsed\n", cycles_per);

    return 0;
}
"""

def compile_run(code, core):
    with tempfile.TemporaryDirectory() as tmpdir:
        asm_file = str(Path(tmpdir) / "routine.s")
        c_file = str(Path(tmpdir) / "routine.c")
        o_file = str(Path(tmpdir) / "routine.o")
        exe_file = str(Path(tmpdir) / "routine")
        with open(asm_file, "w") as f:
            if core.isa == X86: # No need on ARM, assembler can manage constant pool
                print(".intel_syntax noprefix", file=f)
                print("    .section .note.GNU-stack,\"\",@progbits", file=f)
                print(".section .rodata", file=f)
                print("ABS_MASK:", file=f)
                print("  .quad 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF", file=f)
                print("LOOP_ITERS:", file=f)
                print(f"  .quad {ASM_ITERATIONS}", file=f)
                print(file=f)

            Listing(lambda code: (), {}).to_asm(core.isa, "null_loop", f)
            code.to_asm(core.isa, "bench_loop", f)
        with open(c_file, "w") as f:
            f.write(DRIVER)
        if VERBOSE: print(open(asm_file).read())
        subprocess.run(["clang", "-c", asm_file, "-o", o_file])
        #if VERBOSE: subprocess.run(["objdump", "-d", o_file])
        subprocess.run(["clang", c_file, o_file, "-o", exe_file])
        res = subprocess.run([exe_file], stdout=subprocess.PIPE)
    out = res.stdout.decode("ascii").strip().split()
    assert len(out) == 3 and out[1] == "cycles" and out[2] == "elapsed"
    return float(out[0]) / ASM_ITERATIONS

def get_code(name):
    if "[" in name:
        name, rest = name.split("[")
        assert rest.endswith("]")
        rest = rest.removesuffix("]")
        args = rest.split(",")
        kwargs = {}
        for arg in args:
            key, value = arg.split("=")
            if value.isdigit():
                value = int(value)
            else:
                value = get_code(value)
            kwargs[key] = value
        return get_code(name).set_kwargs(kwargs)
    else:
        return CODES[name]

def main():
    import argparse
    # Set up argument parsing
    codes_help = "\n".join(sorted(CODES.keys()))
    parser = argparse.ArgumentParser(
        description="Simulate CPU performance for TwoSum variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available codes:\n{codes_help}"
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--core', type=str, default="P", choices=CORES.keys(), help='P-core or E-core')
    parser.add_argument('--instances', type=int, default=1, help='Number of instances')
    parser.add_argument('--mode', choices=['simulate', 'measure', 'both'], default='both',
                        help='Run simulation, measurement, or both')
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
        metric = "throughput" if num_instances > 1 else "latency"
        results = []
        if args.mode != "measure":
            sim = CPU(core, code).simulate()
            results.append(f"{sim:.2f} simulated {metric}")
        if args.mode != "simulate":
            meas = compile_run(code, core)
            results.append(f"{meas:.2f} measured {metric}")
        print(f"{name:>20}: {', '.join(results)}")

if __name__ == "__main__":
    main()
