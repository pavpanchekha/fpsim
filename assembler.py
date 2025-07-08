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
        if not signature: # Can override for uops
            signature = self.isa.instructions[name]
        argbuf = list(args)
        out = []
        for arg in signature.args:
            flags = arg.split()
            if "const" in flags:
                assert argbuf, f"{name} missing a constant argument"
                argval = argbuf.pop(0)
                assert isinstance(argval, str), f"{name} argument {argval!r} not a constant"
            if "in" in flags:
                assert argbuf, f"{name} missing a register argument"
                argval = argbuf.pop(0)
                assert isinstance(argval, int), f"{name} argument {argval!r} not a register"
                if "flags" in flags:
                    assert argval == self.curflags, \
                        f"{name} argument {argval!r} is not current flags"
            if "out" in flags:
                out.append(self.mkreg())
                if "flags" in flags:
                    self.curflags = out[-1]
        assert len(out) == 1, f"{name} has {len(out)} output registers"
        self.code.append([out[0], name, args])
        return out[0]

    def exec(self, f, *args, **kwargs):
        f(self, *args, **kwargs)
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.push_instruction(name, args, **kwargs)
