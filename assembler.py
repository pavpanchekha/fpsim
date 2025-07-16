from __future__ import annotations
import typing
from dataclasses import dataclass
from typing import TypeAlias

# Basic aliases used throughout the simulator
Register: TypeAlias = int
Constant: TypeAlias = str
Instruction: TypeAlias = tuple[Register, str, list[Register | Constant]]

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
        "v2d": "v{}.2d",
        "v8b": "v{}.8b",
        "flags": None,
    },
    instructions={
        "fadd": Sig("out", "in", "in"),
        "fsub": Sig("out", "in", "in"),
        "fabs": Sig("out", "in"),
        "fcmp": Sig("out flags", "in", "in"),
        "fcsel": Sig("out", "in flags", "in", "in", suffix="lt"),
        "fmov": Sig("out", "in"),
        "fcmge": Sig("out v2d", "in v2d", "in v2d"),
        "bsl": Sig("in out v8b", "in v8b", "in v8b"),
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
        self.code: list[Instruction] = []
        self.flags: set[int] = set()
        self.curflags: typing.Optional[int] = None

    def mkreg(self) -> Register:
        out = self.regs
        self.regs += 1
        return out

    def push_instruction(
        self,
        name: str,
        args: typing.Sequence[Register | Constant],
        signature: Sig | None = None,
    ) -> Register:
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
        self.code.append((out[0], name, list(args)))
        return out[0]

    def exec(
        self,
        f: typing.Callable[..., typing.Any],
        *args: Register | Constant,
        **kwargs: typing.Any,
    ) -> "Assembler":
        f(self, *args, **kwargs)
        return self

    def __getattr__(self, name: str) -> typing.Callable[..., Register]:
        return lambda *args, **kwargs: self.push_instruction(name, args, **kwargs)
