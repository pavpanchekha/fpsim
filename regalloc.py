import time
import pulp

from assembler import Assembler

def compute_live_ranges(asm):
    code = asm.code
    isa = asm.isa

    regs: set[int] = set()
    regstart: dict[int, int] = {}
    regend: dict[int, int] = {}

    for pc, (out, _op, args) in enumerate(code):
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
    
    return regs, regstart, regend

def allocate_registers_ilp(asm: Assembler, verbose = False):
    regs, regstart, regend = compute_live_ranges(asm)
    iargs = {r for r in regs if r < 0}

    if verbose:
        print("Performing register allocation with PuLP")
    start = time.time()
    model = pulp.LpProblem(sense=pulp.LpMinimize)
    at = {}
    for vreg in regs:
        for preg in range(asm.isa.registers):
            at[(vreg, preg)] = pulp.LpVariable(
                f"at_{vreg}_{preg}", cat=pulp.LpBinary
            )

    for vreg in regs:
        if vreg in asm.flags:
            continue
        model += sum(at[(vreg, p)] for p in range(asm.isa.registers)) == 1

    for r1 in regs:
        for r2 in regs:
            if r1 in asm.flags or r2 in asm.flags:
                continue
            if r1 >= r2:
                continue
            if regstart[r1] < regend[r2] and regstart[r2] < regend[r1]:
                for p in range(asm.isa.registers):
                    model += at[(r1, p)] + at[(r2, p)] <= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=100))
    assert model.status == pulp.LpStatusOptimal, "Too many registers, could not allocate"

    assignment = {}
    for vreg in regs:
        for preg in range(asm.isa.registers):
            if at[(vreg, preg)].value() == 1:
                assignment[vreg] = preg
                break
    if verbose:
        elapsed = (time.time() - start) * 1000
        used = len(set(assignment.values()))
        print(f"  Done in {elapsed:.02f}ms with {used} registers")

    return iargs, assignment

class NotEnoughRegistersError(Exception): pass

def allocate_registers_simple(asm : Assembler, verbose=False):
    regs, regstart, regend = compute_live_ranges(asm)
    iargs = {r for r in regs if r < 0}

    reg_status = { i: False for i in range(asm.isa.registers) }
    assignment = {}
    max_regs = 0
    for out in sorted(iargs):
        for reg in reg_status:
            if reg_status[reg] == False:
                reg_status[reg] = out
                assignment[out] = reg
                if verbose: print(f"Assign {out} to {reg}")
                break
    for pc, (out, _op, args) in enumerate(asm.code):
        if verbose: print(f"%{pc}: {out} = {_op} {args}")
        for arg in args:
            assert arg in assignment, f"Argument {arg} not in a register"
            assert reg_status[assignment[arg]] is not False, f"Argument {arg} register {assignment[arg]} is clear"
        for reg in reg_status:
            if reg_status[reg] is not False and regend[reg_status[reg]] < pc:
                reg_status[reg] = False
                if verbose: print(f"Clear {reg}")
        if out in assignment:
            assert reg_status[assignment[out]] is False or assignment[out] == assignment[args[0]]
        else:
            for reg in reg_status:
                if reg_status[reg] is False:
                    reg_status[reg] = out
                    assignment[out] = reg
                    if verbose: print(f"Assign {out} to {reg}")
                    break
            else:
                if verbose: print(f"Count not place {out}")
                raise NotEnoughRegistersError
        max_regs = max(max_regs, len([i for i in reg_status if reg_status[i] is not False]))
    if verbose: print(f"Done, max {max_regs} regs")
    return iargs, assignment

def allocate_registers(asm, verbose=False):
    try:
        iargs, assignment = allocate_registers_simple(asm, verbose)
    except NotEnoughRegistersError:
        iargs, assignment = allocate_registers_ilp(asm, verbose)
    
    allocated: list[tuple[int, str, list[int | str]]] = [
        (assignment[out], op, [assignment[a] if isinstance(a, int) else a for a in args])
        for out, op, args in asm.code
    ]
    iargs_phys = {assignment[r] for r in iargs}
    return allocated, iargs_phys
