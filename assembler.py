from __future__ import annotations
import typing
from dataclasses import dataclass

@dataclass
class Sig:
    args: list[str]
    suffix: typing.Union[None, str]

    def __init__(self, *args, suffix=None):
        self.args = list(args)
        self.suffix = suffix

@dataclass
class ISA:
    registers: int
    prefix: str
    instructions: dict[str, Sig]

ARM = ISA(
    registers=32,
    prefix="d",
    instructions={
        "fadd": Sig("out", "in", "in"),
        "fsub": Sig("out", "in", "in"),
        "fabs": Sig("out", "in"),
        "fcmp": Sig("out flags", "in", "in"),
        "fcsel": Sig("out", "in flags", "in", "in", suffix="lt"),
    },
)

X86 = ISA(
    registers=16,
    prefix="xmm",
    instructions={
        "vaddsd": Sig("out", "in", "in"),
        "vsubsd": Sig("out", "in", "in"),
        "vandpd": Sig("out", "in", "const"),
        "vcmpltsd": Sig("out", "in", "in"),
        "vpcmpgtq": Sig("out", "in", "in"),
        "vblendvpd": Sig("out", "in", "in", "in"),
    },
)

class Assembler:
    def __init__(self, isa: ISA):
        self.isa = isa
        self.ip = 0
        self.code = []
        self.flags: set[int] = set()
        self.curflags: typing.Optional[int] = None

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
