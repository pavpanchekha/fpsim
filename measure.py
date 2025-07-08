# Measurement utilities extracted from main.py
import types
import tempfile
from pathlib import Path
import subprocess
import time
import pulp
import assembler
from assembler import ARM, X86, ISA

VERBOSE = False

ASM_ITERATIONS = 100_000_000

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


def to_asm(listing, isa: ISA, name, fd):
    """Assemble *listing* for the given ISA and write it to *fd*."""
    l = assembler.Assembler(isa)
    iargs = range(-listing.argnum, 0)
    listing.f(l, *iargs)

    regs = set()
    regstart = {}
    regend = {}
    for arg in iargs:
        regstart[arg] = -1  # before start
        regs.add(arg)
    for out, op, args in l.code:
        if out not in iargs:
            regstart[out] = out
            regend[out] = out
        regs.add(out)
        for arg in args:
            if isinstance(arg, str):
                continue
            regend[arg] = max(regend.get(arg, out), out)

    if VERBOSE:
        print("Performing register allocation with PuLP")
    start_time = time.time()
    model = pulp.LpProblem(sense=pulp.LpMinimize)

    at = {}
    for vreg in regs:
        for preg in range(isa.registers):
            at[(vreg, preg)] = pulp.LpVariable(f"at_{vreg}_{preg}", cat=pulp.LpBinary)

    for vreg in regs:
        if vreg in l.flags:
            continue  # Flags don't take up an ISA register
        model += sum(at[vreg, preg] for preg in range(isa.registers)) == 1

    for reg1 in regs:
        for reg2 in regs:
            if reg1 == reg2:
                break
            if regstart[reg1] < regend[reg2] and regstart[reg2] < regend[reg1]:
                for preg in range(isa.registers):
                    model += at[(reg1, preg)] + at[(reg2, preg)] <= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
    assert model.status == pulp.LpStatusOptimal, "Too many registers, could not allocate"

    assignment = {}
    for vreg in regs:
        for preg in range(isa.registers):
            if at[(vreg, preg)].value() == 1:
                assignment[vreg] = preg
    if VERBOSE:
        print(
            "  Done in {:.02f}ms with {} registers".format(
                (time.time() - start_time) * 1000,
                len(set(assignment.values())),
            )
        )

    print(".text", file=fd)
    print(f".global _{name}, {name}", file=fd)
    print(f"{name}:", file=fd)
    print(f"_{name}:", file=fd)

    if isa == ARM:  # No need on x86, all xmm registers caller-saved
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

    for out, op, args in l.code:
        if op == isa.mov and assignment[out] == assignment[args[0]]:
            continue
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

    if isa == ARM:
        print("  subs x9, x9, #1", file=fd)
        print("  b.ne 1b", file=fd)
    elif isa == X86:
        print("  dec r11", file=fd)
        print("  jnz 1b", file=fd)
    if isa == ARM:
        print("  ldp    q14, q15, [sp], #32", file=fd)
        print("  ldp    q12, q13, [sp], #32", file=fd)
        print("  ldp    q10, q11, [sp], #32", file=fd)
        print("  ldp    q8,  q9,  [sp], #32", file=fd)
    print("  ret", file=fd)


def compile_run(code, core):
    """Compile *code* for *core* and measure its latency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        asm_file = str(Path(tmpdir) / "routine.s")
        c_file = str(Path(tmpdir) / "routine.c")
        o_file = str(Path(tmpdir) / "routine.o")
        exe_file = str(Path(tmpdir) / "routine")
        with open(asm_file, "w") as f:
            if core.isa == X86:  # No need on ARM, assembler can manage constant pool
                print(".intel_syntax noprefix", file=f)
                print("    .section .note.GNU-stack,\"\",@progbits", file=f)
                print(".section .rodata", file=f)
                print("ABS_MASK:", file=f)
                print("  .quad 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF", file=f)
                print("LOOP_ITERS:", file=f)
                print(f"  .quad {ASM_ITERATIONS}", file=f)
                print(file=f)

            null_listing = types.SimpleNamespace(f=lambda code: None, argnum=0)
            to_asm(null_listing, core.isa, "null_loop", f)
            to_asm(code, core.isa, "bench_loop", f)
        with open(c_file, "w") as f:
            f.write(DRIVER)
        if VERBOSE:
            print(open(asm_file).read())
        subprocess.run(["clang", "-c", asm_file, "-o", o_file])
        subprocess.run(["clang", c_file, o_file, "-o", exe_file])
        res = subprocess.run([exe_file], stdout=subprocess.PIPE)
    out = res.stdout.decode("ascii").strip().split()
    assert len(out) == 3 and out[1] == "cycles" and out[2] == "elapsed"
    return float(out[0]) / ASM_ITERATIONS
