import subprocess
import tempfile
from pathlib import Path
from assembler import Assembler, ARM, X86
from regalloc import allocate_registers

ASM_ITERATIONS = 1_000_000

def write_asm(code, isa, name, fd, *, iargs=None):
    """Write ``code`` as assembly for ``isa`` to ``fd``."""
    if iargs is None:
        iargs = set()

    print(".text", file=fd)
    print(f".global _{name}, {name}", file=fd)
    print(f"{name}:", file=fd)
    print(f"_{name}:", file=fd)

    # Prologue
    if isa == ARM:  # No need on x86, all xmm registers caller-saved
        print("  stp    q8,  q9,  [sp, #-32]!", file=fd)
        print("  stp    q10, q11, [sp, #-32]!", file=fd)
        print("  stp    q12, q13, [sp, #-32]!", file=fd)
        print("  stp    q14, q15, [sp, #-32]!", file=fd)

    if isa == ARM:
        for ireg in iargs:
            print(f"  fmov d{ireg}, #0.0", file=fd)
        print(f"  ldr x9, ={ASM_ITERATIONS}", file=fd)
    elif isa == X86:
        for ireg in iargs:
            print(f"  xorpd xmm{ireg}, xmm{ireg}", file=fd)
        print(f"  mov r11, [rip+LOOP_ITERS]", file=fd)

    print(".p2align 4", file=fd)
    print("1:", file=fd)

    for out, op, args in code:
        # Apple M1 hiccups if you fmov a register to itself, don't do it
        if op == isa.mov and out == args[0]:
            continue
        signature = isa.instructions[op]
        arglist = []
        for arg, sig in zip([out] + list(args), signature.args):
            flags = sig.split()
            for fl in flags:
                if fl in isa.prefix:
                    pfx = isa.prefix[fl]
                    break
            else:
                pfx = isa.prefix.get(None)
            if pfx is None:
                # None means implicit register
                continue
            if "const" in flags and isinstance(arg, str):
                arglist.append(f"[rip+{arg}]")
            else:
                arglist.append(pfx.format(arg))
        if signature.suffix:
            arglist.append(signature.suffix)
        print(f"  {op} {', '.join(arglist)}", file=fd)

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

# Written by ChatGPT o4-mini
DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#elif defined(__linux__)
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <string.h>
#include <unistd.h>
static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}
#endif

extern void null_loop(void);
extern void bench_loop(void);

int main(void) {
#ifdef __APPLE__
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
#elif defined(__linux__)
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
int fd = perf_event_open(&pe, 0, -1, -1, 0);
int use_rdtsc = 0;
if (fd == -1) {
    perror("perf_event_open");
    use_rdtsc = 1;
}
#endif

    uint64_t baseline = 0xffffffffull;

    for (int i = 0; i < 100; i++) {
#ifdef __APPLE__
      uint64_t start = mach_absolute_time();
      null_loop();
      uint64_t end = mach_absolute_time();
      if (end - start < baseline) baseline = end - start;
#elif defined(__linux__)
      if (!use_rdtsc) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        null_loop();
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t count;
        read(fd, &count, sizeof(count));
        if (count < baseline) baseline = count;
      } else {
        uint64_t start = __builtin_ia32_rdtsc();
        null_loop();
        uint64_t end = __builtin_ia32_rdtsc();
        if (end - start < baseline) baseline = end - start;
      }
#else
      uint64_t start = __builtin_ia32_rdtsc();
      null_loop();
      uint64_t end = __builtin_ia32_rdtsc();
      if (end - start < baseline) baseline = end - start;
#endif
    }

    double ticks = 1.0e308;
    for (int i = 0; i < 100; i++) {
#ifdef __APPLE__
      uint64_t start = mach_absolute_time();
      bench_loop();
      uint64_t end = mach_absolute_time();
      if (end - start - baseline < ticks) ticks = end - start - baseline;
#elif defined(__linux__)
      if (!use_rdtsc) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        bench_loop();
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t count;
        read(fd, &count, sizeof(count));
        if (count - baseline < ticks) ticks = count - baseline;
      } else {
        uint64_t start = __builtin_ia32_rdtsc();
        bench_loop();
        uint64_t end = __builtin_ia32_rdtsc();
        if (end - start - baseline < ticks) ticks = end - start - baseline;
      }
#else
      uint64_t start = __builtin_ia32_rdtsc();
      bench_loop();
      uint64_t end = __builtin_ia32_rdtsc();
      if (end - start - baseline < ticks) ticks = end - start - baseline;
#endif
    }

#ifdef __APPLE__
    double cycles_per = ticks * timebase.numer / timebase.denom * 3.2;
#elif defined(__linux__)
    if (!use_rdtsc)
        close(fd);
    double cycles_per = ticks;
#else
    double cycles_per = ticks;
#endif
    printf("%.2f cycles elapsed\n", cycles_per);

    return 0;
}
"""

def compile_run(code, core, ssh_host=None, verbose=False):
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

            asm = Assembler(core.isa)
            ncode, ninputs = allocate_registers(asm, verbose=verbose)
            write_asm(ncode, asm.isa, "null_loop", f, iargs=ninputs)

            asm = Assembler(core.isa).exec(code.f, *range(-code.argnum, 0))
            bcode, binputs = allocate_registers(asm, verbose=verbose)
            write_asm(bcode, asm.isa, "bench_loop", f, iargs=binputs)
        with open(c_file, "w") as f:
            f.write(DRIVER)
        if verbose:
            print(open(asm_file).read())
        if ssh_host is None:
            subprocess.run(["clang", "-c", asm_file, "-o", o_file])
            subprocess.run(["clang", c_file, o_file, "-o", exe_file])
            res = subprocess.run([exe_file], stdout=subprocess.PIPE)
        else:
            remote_tmp = subprocess.check_output([
                "ssh", ssh_host, "mktemp", "-d"
            ]).decode().strip()
            subprocess.run([
                "scp", "-q", asm_file, f"{ssh_host}:{remote_tmp}/routine.s"
            ])
            subprocess.run([
                "scp", "-q", c_file, f"{ssh_host}:{remote_tmp}/routine.c"
            ])
            subprocess.run([
                "ssh", ssh_host,
                f"clang -c {remote_tmp}/routine.s -o {remote_tmp}/routine.o"
            ])
            subprocess.run([
                "ssh", ssh_host,
                f"clang {remote_tmp}/routine.c {remote_tmp}/routine.o -o {remote_tmp}/routine"
            ])
            res = subprocess.run([
                "ssh", ssh_host, f"{remote_tmp}/routine"
            ], stdout=subprocess.PIPE)
            subprocess.run(["ssh", ssh_host, "rm", "-rf", remote_tmp])
    out = res.stdout.decode("ascii").strip().split()
    assert len(out) == 3 and out[1] == "cycles" and out[2] == "elapsed"
    return float(out[0]) / ASM_ITERATIONS
