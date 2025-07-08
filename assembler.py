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
    prefix: dict[typing.Optional[str], typing.Optional[str]]
    instructions: dict[str, Sig]
    mov: str

ARM = ISA(
    registers=32,
    prefix={
        None: "d{}",
        "vector": "v{}.2d",
        "flags": None,
    },
    instructions={
        "fadd": Sig("out", "in", "in"),
        "fsub": Sig("out", "in", "in"),
        "fabs": Sig("out", "in"),
        "fcmp": Sig("out flags", "in", "in"),
        "fcsel": Sig("out", "in flags", "in", "in", suffix="lt"),
        "fmov": Sig("out", "in"),
        "fcmge": Sig("out vector", "in vector", "in vector"),
        "bsl": Sig("in out vector", "in vector", "in vector"),
    },
    mov="fmov",
)

X86 = ISA(
    registers=16,
    prefix={
        None: "xmm{}",
        "flags": None,
    },
    instructions={
        "movsd": Sig("out", "in"),
        "vaddsd": Sig("out", "in", "in"),
        "vsubsd": Sig("out", "in", "in"),
        "vandpd": Sig("out", "in", "const"),
        "vcmpltsd": Sig("out", "in", "in"),
        "vpcmpgtq": Sig("out", "in", "in"),
        "vblendvpd": Sig("out", "in", "in", "in"),
    },
    mov="movsd",
)

class Assembler:
    def __init__(self, isa: ISA):
        self.isa = isa
        self.regs = 0
        self.code = []
        self.flags: set[int] = set()
        self.curflags: typing.Optional[int] = None

    def mkreg(self):
        out = self.regs
        self.regs += 1
        return out

    def push_instruction(self, name, args, signature=None):
        out = self.mkreg()
        if not signature: # Can override for uops
            signature = self.isa.instructions[name]
        assert "out" in signature.args[0].split()
        assert len(args) == len(signature.args[1:]), f"{name} wrong number of arguments"
        if "flags" in signature.args[0].split():
            self.flags.add(out)
            self.curflags = out
        for arg, sig in zip(args, signature.args[1:]):
            if "flags" in sig.split():
                assert arg in self.flags, f"{name} argument not a flags register"
                assert arg == self.curflags, f"{name} argument is not current flags"
            elif "const" in sig.split():
                assert isinstance(arg, str), f"{name} argument not a constant"
            elif "in" in sig.split() or "vector" in sig.split():
                assert isinstance(arg, int), f"{name} argument not a register"
        self.code.append([out, name, args])
        return out

    def exec(self, f, *args, **kwargs):
        f(self, *args, **kwargs)
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.push_instruction(name, args, **kwargs)
