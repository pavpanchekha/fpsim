from __future__ import annotations

from dataclasses import dataclass

from assembler import ISA, ARM, X86, Assembler

@dataclass
class Core:
    isa: "ISA"
    latencies: dict[str, tuple[int, list[int]]]
    priority: list[int]



class CPU:
    def __init__(self, core: Core, code, *, verbose: bool = False):
        self.core = core
        self.verbose = verbose
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
                if self.verbose:
                    print(f"[{self.cycle:>5}] {pc} ({opcode}) start on u{unit}")
                self.status[pc] = latency
                units[unit] = True
                return

    def tick(self):
        units = {}
        for instruction in self.code:
            self.dispatch(instruction, units)
        for pc, latency in self.status.items():
            if latency == 0:
                continue
            self.status[pc] = latency - 1
        if all(self.status.get(oarg) == 0 for oarg in self.oargs):
            if self.verbose:
                print(f"Instance completed on cycle {self.cycle}")
            self.completions += 1
            self.status = {}
        self.cycle += 1

    def simulate(self, cycles=10000):
        for _ in range(cycles):
            self.tick()
        return self.cycle / self.completions


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
}, priority=[7, 6])

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
