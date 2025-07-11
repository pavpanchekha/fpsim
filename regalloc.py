import time
import pulp

from assembler import Assembler


def allocate_registers(
    asm: Assembler,
    *,
    verbose: bool = False,
) -> tuple[list[tuple[int, str, list[int | str]]], set[int]]:
    """Allocate virtual registers in ``asm`` to ISA registers.

    Parameters
    ----------
    asm:
        Assembler object whose ``code`` will be allocated.
    verbose:
        Print solver progress information.

    Returns
    -------
    allocated_code:
        New code list with physical register numbers.
    iargs:
        Set of physical registers corresponding to the original input
        arguments (register numbers < -1).
    """
    flags = asm.flags
    code = asm.code
    isa = asm.isa

    regs: set[int] = set()
    regstart: dict[int, int] = {}
    regend: dict[int, int] = {}

    for pc, (out, _op, args) in enumerate(code):
        if isinstance(out, int):
            if out >= 0:
                regstart[out] = min(regstart.get(out, pc), pc)
                regend[out] = pc
            regs.add(out)
        for arg in args:
            if not isinstance(arg, int):
                continue
            if arg < 0:
                regstart[arg] = -1
            regend[arg] = max(regend.get(arg, pc), pc)
            regs.add(arg)

    iargs = {r for r in regs if r < -1}

    if regs:
        if verbose:
            print("Performing register allocation with PuLP")
        start = time.time()
        model = pulp.LpProblem(sense=pulp.LpMinimize)
        at = {}
        for vreg in regs:
            for preg in range(isa.registers):
                at[(vreg, preg)] = pulp.LpVariable(
                    f"at_{vreg}_{preg}", cat=pulp.LpBinary
                )

        for vreg in regs:
            if vreg in flags:
                continue
            model += sum(at[(vreg, p)] for p in range(isa.registers)) == 1

        for r1 in regs:
            for r2 in regs:
                if r1 >= r2:
                    continue
                if regstart.get(r1, 0) < regend.get(r2, 0) and regstart.get(r2, 0) < regend.get(r1, 0):
                    for p in range(isa.registers):
                        model += at[(r1, p)] + at[(r2, p)] <= 1

        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        assert model.status == pulp.LpStatusOptimal, "Too many registers, could not allocate"

        assignment = {}
        for vreg in regs:
            for preg in range(isa.registers):
                if at[(vreg, preg)].value() == 1:
                    assignment[vreg] = preg
                    break
        if verbose:
            elapsed = (time.time() - start) * 1000
            used = len(set(assignment.values()))
            print(f"  Done in {elapsed:.02f}ms with {used} registers")
    else:
        assignment = {}

    allocated: list[tuple[int, str, list[int | str]]] = [
        (assignment[out], op, [assignment[a] if isinstance(a, int) else a for a in args])
        for out, op, args in code
    ]
    iargs_phys = {assignment[r] for r in iargs}
    return allocated, iargs_phys
