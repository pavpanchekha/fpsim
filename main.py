from dataclasses import dataclass
from functools import wraps
import inspect
import pulp
import tempfile
from pathlib import Path
import subprocess
import sys
import time

REGISTERS = 32

PCORE = {
    "fadd": (3, [11, 12, 13, 14]),
    "fsub": (3, [11, 12, 13, 14]),
    "fabs": (2, [11, 12, 13, 14]),
    "fcmp": (2, [11]),
    "fcsel": (2, [13, 14]),
}

ECORE = {
    "fadd": (3, [6, 7]),
    "fsub": (3, [6, 7]),
    "fabs": (2, [6, 7]),
    "fcmp": (2, [6]),
    "fcsel": (2, [6, 7]),
}

UNIT_ORDER = [0, 1, 5, 7, 6, 12, 13, 14, 11]

# Coffee Lake
CL_CORE = {
    "vaddsd": (4, [0, 1]),
    "vsubsd": (4, [0, 1]),
    "vandpd": (1, [0, 1, 5]),
    "vcmpltsd": (4, [0, 1]),
    "vpcmpgtq": (3, [5]),
    "vblendvpd": (2, [0, 1, 5]),
}

CORES = {
    "E": ECORE,
    "P": PCORE,
    "CL": CL_CORE,
}

CODES = {}

@dataclass
class Assembler:
    ip: int
    code: list[tuple[int, str, list[int]]]

    def __init__(self, language="arm"):
        self.language = language
        self.ip = 0
        self.code = []
        self.flags = set()
        self.curflags = None

    def push_instruction(self, name, args):
        out = self.ip
        self.ip += 1
        self.code.append([out, name, args])
        if name == "fcmp":
            self.flags.add(out)
            self.curflags = out
        elif name == "fcsel":
            assert args[0] in self.flags, "fcsel argument is not a flags register"
            assert args[0] == self.curflags, "fcsel argument is not current flags"
        return out

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
    def __init__(self, f, kwargs):
        self.f = f
        self.argnum = num_args(f) - 1
        self.kwargs = kwargs

    def code(self, core):
        l = Assembler("arm" if "fadd" in core else "x86")
        self.f(l, *range(-self.argnum, 0), **self.kwargs)
        return l.code

    def __call__(self, code, *args):
        return self.f(code, *args, **self.kwargs)

    def set_kwargs(self, kwargs):
        return Listing(self.f, kwargs)

    def to_asm(self, core, name, fd, instances=1):
        l = Assembler("arm" if "fadd" in core else "x86")
        iargs = []
        oargs = []
        latency_ties = []
        for instance in range(instances):
            this_iargs = list(range(-self.argnum - len(iargs), -len(iargs)))
            this_oargs = self.f(l, *this_iargs, **self.kwargs)
            iargs.extend(this_iargs)
            oargs.extend(this_oargs)
            # We tie all inputs to the last output register
            latency_ties.extend([(iarg, max(*this_oargs)) for iarg in this_iargs])

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

        if VERBOSE: print("Performing register allocation with PuPL")
        start = time.time()
        model = pulp.LpProblem(sense=pulp.LpMinimize)

        REGISTERS = 32 if l.language == "arm" else 16 # TODO: Abstract

        at = {}
        for vreg in regs:
            for preg in range(REGISTERS):
                at[(vreg, preg)] = pulp.LpVariable(f"at_{vreg}_{preg}", cat=pulp.LpBinary)

        for vreg in regs:
            if vreg in l.flags: continue # Flags don't take up an ISA register
            # Each v-reg is in exactly one p-reg
            model += sum(at[vreg, preg] for preg in range(REGISTERS)) == 1

        for reg1 in regs:
            for reg2 in regs:
                if reg1 == reg2: break
                if regstart[reg1] < regend[reg2] and regstart[reg2] < regend[reg1]:
                    for preg in range(REGISTERS):
                        # Can't put two interfering v-regs in the same p-reg
                        model += at[(reg1, preg)] + at[(reg2, preg)] <= 1

        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        assert model.status == pulp.LpStatusOptimal, "Too many registers, could not allocate"

        assignment = {}
        for vreg in regs:
            for preg in range(REGISTERS):
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
        if l.language == "arm": # No need on x86, all xmm registers caller-saved
            print("  stp    q8,  q9,  [sp, #-32]!", file=fd)
            print("  stp    q10, q11, [sp, #-32]!", file=fd)
            print("  stp    q12, q13, [sp, #-32]!", file=fd)
            print("  stp    q14, q15, [sp, #-32]!", file=fd)

        if l.language == "arm":
            for ireg in iargs:
                print(f"  fmov d{assignment[ireg]}, #0.0", file=fd)
            print(f"  ldr x9, ={ASM_ITERATIONS}", file=fd) # Loop 1M times
        elif l.language == "x86":
            for ireg in iargs:
                print(f"  xorpd xmm{assignment[ireg]}, xmm{assignment[ireg]}", file=fd)
            print(f"  mov r11, [rip+LOOP_ITERS]", file=fd)

        print(".p2align 4", file=fd)
        print("1:", file=fd)

        PREFIX = "d" if l.language == "arm" else "xmm"
        for out, op, args in l.code:
            if op == "fcmp":
                argstr = ", ".join(f"{PREFIX}{assignment[arg]}" for arg in args)
                print(f"  {op} {argstr}", file=fd) # Don't use flag register
            elif op == "fcsel":
                argstr = ", ".join(f"{PREFIX}{assignment[arg]}" for arg in args if arg not in l.flags)
                print(f"  {op} {PREFIX}{assignment[out]}, {argstr}, gt", file=fd)
            elif op == "vandpd":
                arglist = []
                for arg in args:
                    if isinstance(arg, int):
                        arglist.append(PREFIX + str(assignment[arg]))
                    elif isinstance(arg, str):
                        arglist.append(f"[rip+{arg}]")
                argstr = ", ".join(arglist)
                print(f"  {op} {PREFIX}{assignment[out]}, {argstr}", file=fd)
            else:
                argstr = ", ".join(f"{PREFIX}{assignment[arg]}" for arg in args)
                print(f"  {op} {PREFIX}{assignment[out]}, {argstr}", file=fd)

        # At the tail we need to move output registers back into input
        # registers to enforce loop-carried dependency.
        if l.language == "arm":
            for dst, src in latency_ties:
                if assignment[dst] == assignment[src]: continue
                print(f"  fmov d{assignment[dst]}, d{assignment[src]}", file=fd)
        elif l.language == "x86":
            for dst, src in latency_ties:
                if assignment[dst] == assignment[src]: continue
                print(f"  movsd xmm{assignment[dst]}, xmm{assignment[src]}", file=fd)
            
        if l.language == "arm":
            print("  subs x9, x9, #1", file=fd)
            print("  b.ne 1b", file=fd)
        elif l.language == "x86":
            print("  dec r11", file=fd)
            print("  jnz 1b", file=fd)
            
        if l.language == "arm": # No need on x86, all xmm caller-saved
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
    if code.language == "arm":
        s = code.fadd(a, b)
        return s, code.fsub(b, code.fsub(s, a))
    elif code.language == "x86":
        s = code.vaddsd(a, b)
        return s, code.vsubsd(b, code.vsubsd(s, a))

@algorithm
def ts(code, a, b):
    if code.language == "arm":
        s = code.fadd(a, b)
        bb = code.fsub(s, a)
        return s, code.fadd(code.fsub(a, code.fsub(s, bb)), code.fsub(b, bb))
    elif code.language == "x86":
        s = code.vaddsd(a, b)
        bb = code.vsubsd(s, a)
        return s, code.vaddsd(code.vsubsd(a, code.vsubsd(s, bb)), code.vsubsd(b, bb))

def sort2(code, a, b):
    if code.language == "arm":
        aa, ab = code.fabs(a), code.fabs(b)
        cond = code.fcmp(aa, ab)
        x = code.fcsel(cond, a, b)
        y = code.fcsel(cond, b, a)
    elif code.language == "x86":
        aa, ab = code.vandpd(a, "ABS_MASK"), code.vandpd(b, "ABS_MASK")
        cond = code.vcmpltsd(aa, ab)
        x = code.vblendvpd(a, b, cond)
        y = code.vblendvpd(b, a, cond)
    return x, y

def add(code, a, b):
    if code.language == "arm":
        return code.fadd(a, b)
    elif code.language == "x86":
        return code.vaddsd(a, b)

def sub(code, a, b):
    if code.language == "arm":
        return code.fsub(a, b)
    elif code.language == "x86":
        return code.vsubsd(a, b)

def fabs(code, a):
    if code.language == "arm":
        return code.fabs(a)
    elif code.language == "x86":
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
    if code.language == "arm":
        cond = code.fcmp(code.fabs(b), code.fabs(a))
        x = code.fcsel(cond, a, b)
        xx = code.fcsel(cond, aa, bb)
    elif code.language == "x86":
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
    if code.language == "arm":
        cond = code.fcmp(aa, bb)
        return s, code.fcsel(cond, ifa, ifb)
    elif code.language == "x86":
        cond = code.vcmpltsd(aa, bb)
        return s, code.vblendvpd(ifa, ifb, cond)

TS=ts

@algorithm(ts=TS)
def ddadd(code, x0, y0, x1, y1, *, ts=ts):
    x0, y0 = ts(code, x0, y0)
    x1, y1 = ts(code, x1, y1)
    y0     = add(code, y0, x1)
    x0, y0 = ts(code, x0, y0)
    y0     = add(code, y0, y1)
    x0, y0 = ts(code, x0, y0)
    return x0, y0

@algorithm(ts=TS, ts2=fts)
def madd(code, x0, y0, x1, y1, *, ts=ts, ts2=fts):
    x0, y0 = ts(code, x0, y0)
    x1, y1 = ts(code, x1, y1)
    x0, x1 = ts(code, x0, x1)
    y0     = add(code, y0, y1)
    y0     = add(code, y0, x1)
    x0, y0 = ts2(code, x0, y0)
    return x0, y0

VERBOSE = False

class CPU:
    def __init__(self, core, code, instances):
        self.core = core
        self.code = code
        self.instances = instances

        self.last = max(pc for pc, opcode, deps in code)
        self.status = {}
        
        self.cycle = 0
        self.completions = 0

    def dispatch(self, instance, instruction, units):
        pc, opcode, deps = instruction
        latency, ports = self.core[opcode]
        if (instance, pc) in self.status:
            return None
        if not all(self.status.get((instance, dep)) == 0 for dep in deps if isinstance(dep, int) and dep >= 0):
            return None
        for unit in UNIT_ORDER:
            if unit not in units and unit in ports:
                if VERBOSE: print(f"[{self.cycle:>5}] {instance}%{pc} ({opcode}) start on u{unit}")
                self.status[(instance, pc)] = latency
                units[unit] = True
                return

    def tick(self):
        units = {}
        for instance in range(self.instances):
            for instruction in reversed(self.code):
                self.dispatch(instance, instruction, units)
        for (instance, pc), latency in self.status.items():
            if latency == 0: continue
            self.status[(instance, pc)] = latency - 1
        for instance in range(self.instances):
            if self.status.get((instance, self.last)) == 0:
                if VERBOSE: print(f"Instance {instance} completed on cycle {self.cycle}")
                self.completions += 1
                for instruction in self.code:
                    del self.status[(instance, instruction[0])]
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
    uint64_t baseline = 0;//end - start;

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

def compile_run(code, core, instances=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        asm_file = str(Path(tmpdir) / "routine.s")
        c_file = str(Path(tmpdir) / "routine.c")
        o_file = str(Path(tmpdir) / "routine.o")
        exe_file = str(Path(tmpdir) / "routine")
        with open(asm_file, "w") as f:
            if "fadd" not in core: # No need on ARM, assembler can manage constant pool
                print(".intel_syntax noprefix", file=f)
                print("    .section .note.GNU-stack,\"\",@progbits", file=f)
                print(".section .rodata", file=f)
                print("ABS_MASK:", file=f)
                print("  .quad 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF", file=f)
                print("LOOP_ITERS:", file=f)
                print(f"  .quad {ASM_ITERATIONS}", file=f)
                print(file=f)

            Listing(lambda code: (), {}).to_asm(core, "null_loop", f)
            code.to_asm(core, "bench_loop", f, instances=instances)
        with open(c_file, "w") as f:
            f.write(DRIVER)
        if VERBOSE: print(open(asm_file).read())
        subprocess.run(["clang", "-c", asm_file, "-o", o_file])
        #if VERBOSE: subprocess.run(["objdump", "-d", o_file])
        subprocess.run(["clang", c_file, o_file, "-o", exe_file])
        res = subprocess.run([exe_file], stdout=subprocess.PIPE)
    out = res.stdout.decode("ascii").strip().split()
    assert len(out) == 3 and out[1] == "cycles" and out[2] == "elapsed"
    return float(out[0])

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
    parser = argparse.ArgumentParser(description="Simulate CPU performance for TwoSum variants.")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--core', type=str, default="P", choices=CORES.keys(), help='P-core or E-core')
    parser.add_argument('--instances', type=int, default=1, help='Number of instances')
    parser.add_argument('codes', nargs=argparse.REMAINDER, help='Functions to simulate')
    args = parser.parse_args()

    # Assign flags
    global VERBOSE
    VERBOSE = args.verbose
    num_instances = args.instances
    codes = args.codes if args.codes else CODES.keys()

    # Simulate and print results for each core code
    for name in codes:
        core = CORES[args.core]
        code = get_code(name)
        measured = compile_run(code, core, args.instances) / ASM_ITERATIONS / args.instances
        simulated = CPU(core, code.code(core), args.instances).simulate()
        metric = "throughput" if num_instances > 1 else "latency"
        print(f"{name:>20}: {simulated:.2f} simulated {metric}, {measured:.2f} measured {metric}")

if __name__ == "__main__":
    main()
