from __future__ import annotations

from dataclasses import dataclass, field
import typing

from assembler import ISA, ARM, X86, Sig, Assembler, Instruction, Register

@dataclass
class Core:
    isa: "ISA"
    latencies: dict[str, tuple[int, list[int]]]
    priority: list[int]
    microcode: dict[str, typing.Callable] = field(default_factory=dict)

# Apple M1
PCORE = Core(ARM, {
    "fadd": (3, [11, 12, 13, 14]),
    "fsub": (3, [11, 12, 13, 14]),
    "fabs": (2, [11, 12, 13, 14]),
    "fcmp": (2, [11]),
    "fcsel": (2, [13, 14]),
    "fcmge": (2, [11, 12, 13, 14]),
    "bsl": (2, [11, 12, 13, 14]),
}, priority=[12, 13, 14, 11])

ECORE = Core(ARM, {
    "fadd": (3, [6, 7]),
    "fsub": (3, [6, 7]),
    "fabs": (2, [6, 7]),
    "fcmp": (2, [6]),
    "fcsel": (2, [6, 7]),
    "fcmge": (2, [6, 7]),
    "bsl": (2, [6, 7]),
}, priority=[7, 6])

# Coffee Lake
def vblendvpd(code, a, b, mask):
    q = code.vblendvpd_1(mask, signature=Sig("out", "in"))
    return code.vblendvpd_2(a, b, q, signature=Sig("out", "in", "in", "in"))

CL_CORE = Core(X86, {
    "vaddsd": (4, [0, 1]),
    "vsubsd": (4, [0, 1]),
    "vandpd": (1, [0, 1, 5]),
    "vcmpltsd": (4, [0, 1]),
    "vpcmpgtq": (3, [5]),
    "vblendvpd_1": (1, [0, 1, 5]),
    "vblendvpd_2": (1, [0, 1, 5]),
}, microcode={
    "vblendvpd": vblendvpd
}, priority=[0, 1, 5])

CORES = {
    "E": ECORE,
    "P": PCORE,
    "CL": CL_CORE,
}

@dataclass
class Schedule:
    busy : bool = False
    capacity : int = 32
    waiting: list[Instruction] = field(default_factory=list)

class CPU:
    def __init__(self, core: Core, code, *, verbose: bool = False):
        self.core = core
        self.verbose = verbose

        # Front-end
        self.pc = 0
        self.rat = { i: i for i in range(-code.argnum, 0) }
        self.code: list[Instruction] = Assembler(core.isa).exec(code, *range(-code.argnum, 0)).code

        # Back-end
        self.uops = Assembler(self.core.isa)
        self.decode_queue: list[Instruction] = []
        self.units = {p: Schedule() for p in self.core.priority}
        self.metrics: dict[str, list[str]] = {}
        self.inflight: dict[int, int] = {}
        self.done = set(self.rat) # Arguments start complete

        # Perf counters
        self.cycle = 0
        self.retired = 0

    def decode(self, inst: Instruction) -> list[Instruction]:
        out, op, args = inst
        if op == self.core.isa.mov:
            assert len(args) == 1, f"Weird {op} instruction with arguments {args}"
            assert isinstance(args[0], int)
            self.rat[out] = self.rat[args[0]]
        else:
            args = [self.rat[a] if isinstance(a, int) else a for a in args]
            if op in self.core.microcode:
                self.rat[out] = self.core.microcode[op](self.uops, *args)
            else:
                self.rat[out] = self.uops.push_instruction(op, args)
        uops = self.uops.code
        self.uops.code = []
        if self.verbose: print(f"[{self.cycle:>5}] decode {inst} ->", uops)
        return uops

    def advance_pc(self):
        out, op, args = self.code[self.pc]
        self.pc += 1
        if self.pc >= len(self.code):
            self.pc = 0 # Restart the loop

    def retire(self):
        done = []
        for pc, latency in self.inflight.items():
            if latency > 0:
                self.inflight[pc] -= 1
                if self.inflight[pc] == 0:
                    if self.verbose: print(f"[{self.cycle:>5}] retired {pc}")
                    self.retired += 1
                    done.append(pc)
                    self.done.add(pc)
        for pc in done:
            del self.inflight[pc]

    def frontend(self):
        for _ in range(8):
            if len(self.decode_queue) >= 12:
                self.metrics.setdefault("frontend", []).append("full")
                break  # backpressure
            inst = self.code[self.pc]
            self.decode_queue.extend(self.decode(inst))
            self.advance_pc()

    def dispatch(self):
        for _ in range(4):
            if not self.decode_queue:
                break  # done
            out, op, args = self.decode_queue[0]
            bestport, bestunit, bestscore = None, None, None
            for port in self.core.priority:
                if port not in self.core.latencies[op][1]:
                    continue
                unit = self.units[port]
                if unit.busy:
                    continue
                if len(unit.waiting) >= unit.capacity:
                    continue
                if not bestscore or len(unit.waiting) < bestscore:
                    bestport = port
                    bestunit = unit
                    bestscore = len(unit.waiting)
            if not bestunit:
                self.metrics.setdefault("frontend", []).append("stall")
                break  # Backpressure
            # Select the least-busy unit; tie-breaks to highest priority
            if self.verbose:
                print(f"[{self.cycle:>5}] Steering {op} to unit {bestport}")
            bestunit.busy = True
            bestunit.waiting.append((out, op, args))
            self.decode_queue.pop(0)

    def schedule(self):
        for port, unit in self.units.items():
            if len(unit.waiting) >= unit.capacity:
                self.metrics.setdefault(f"p{port}", []).append("full")
            for out, op, args in unit.waiting:
                latency, ports = self.core.latencies[op]
                for arg in args:
                    if arg not in self.done and isinstance(arg, int):
                        break
                else:
                    if self.verbose:
                        print(f"[{self.cycle:>5}] {op} start on u{port}")
                    self.metrics.setdefault(f"p{port}", []).append("start")
                    self.inflight[out] = latency
                    unit.waiting.remove((out, op, args))
                    break  # This port found its task for this cycle
            unit.busy = False  # Reset for next cycle

    def tick(self):
        self.retire()
        self.frontend()
        self.dispatch()
        self.schedule()
        self.cycle += 1

    def simulate(self, cycles=10000):
        for _ in range(cycles):
            self.tick()
        for key, events in sorted(self.metrics.items()):
            evt_types = set(events)
            print(f"{key:>10}:", end="")
            for evt_type in sorted(evt_types):
                pct = events.count(evt_type) / self.cycle if self.cycle else 0
                item = f"{evt_type} ({pct*100:.1f}%)"
                print(f" {item:>16}", end="")
            print()
        true_instructions = len([op for out, op, args in self.code if op != self.core.isa.mov])
        return self.cycle / (self.retired / true_instructions)

