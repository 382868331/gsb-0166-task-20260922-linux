"""无流水顺序参考解释器（独立实现，不导入 pipeline）。

一条指令完整执行（取指/译码/执行/访存/写回一气呵成）后才执行下一条。
它只与 pipeline 共享 :mod:`tinycpu.isa` 的静态定义（指令格式、校验、
32 位常量），期望状态由这里独立推导，用于交叉验证流水线的退休结果。

精确异常语义对齐：遇到非法 LOAD/STORE 时该指令不产生任何架构效果
（LOAD 不写寄存器、STORE 不写内存），机器停在故障点，此前指令全部提交。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .isa import (
    MASK32,
    Fault,
    Instruction,
    Op,
    validate_program,
)
from .mmio import MMIO_READ, MMIO_WRITE, MMIOBus, MMIOEmptyError


@dataclass(frozen=True, slots=True)
class SeqTraceEntry:
    """顺序执行一条指令的退休记录。字段含义与 pipeline.RetireRecord 对齐。"""

    pc: int
    op: Op
    fault: Optional[Fault] = None
    rd: Optional[int] = None
    value: Optional[int] = None
    addr: Optional[int] = None
    mem_value: Optional[int] = None
    mmio: Optional[str] = None  # "read"/"write"：本条在提交时访问了设备


@dataclass(frozen=True, slots=True)
class SeqResult:
    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[SeqTraceEntry, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def run_sequential(
    instructions: list[Instruction] | tuple[Instruction, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
    mmio: Optional[MMIOBus] = None,
) -> SeqResult:
    """顺序解释执行直到 HALT 或非法访存。

    与 :func:`tinycpu.pipeline.run` 使用完全相同的静态校验，
    因此两者对同一程序要么都接受、要么都拒绝。
    """
    if mmio is not None and not isinstance(mmio, MMIOBus):
        raise TypeError(
            f"mmio 必须是 tinycpu.MMIOBus 或 None，收到 {type(mmio).__name__}"
        )
    prog, mem, regs = validate_program(instructions, memory, initial_regs)
    n = len(prog)

    trace: list[SeqTraceEntry] = []
    pc = 0
    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    # 顺序机不会有周期预算问题，但防御失控分支：程序最多走 10_000 步。
    steps = 0
    max_steps = 10_000

    while not halted and fault is None:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("顺序参考执行步数超限（程序可能含无限分支循环）")
        ins = prog[pc]
        op = ins.op

        if op == Op.ADD:
            val = (regs[ins.rs1] + regs[ins.rs2]) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            trace.append(SeqTraceEntry(pc, op, rd=ins.rd if ins.rd else None,
                                       value=val if ins.rd else None))
            pc += 1
        elif op == Op.SUB:
            val = (regs[ins.rs1] - regs[ins.rs2]) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            trace.append(SeqTraceEntry(pc, op, rd=ins.rd if ins.rd else None,
                                       value=val if ins.rd else None))
            pc += 1
        elif op == Op.ADDI:
            val = (regs[ins.rs1] + ins.imm) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            trace.append(SeqTraceEntry(pc, op, rd=ins.rd if ins.rd else None,
                                       value=val if ins.rd else None))
            pc += 1
        elif op == Op.LOAD:
            addr = (regs[ins.rs1] + ins.imm) & MASK32
            if mmio is not None and mmio.contains(addr):
                if not mmio.is_read_reg(addr):
                    fault = Fault.ADDR
                    fault_pc = pc
                    trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
                    break
                try:
                    val = mmio.read()
                except MMIOEmptyError:
                    fault = Fault.MMIO_EMPTY
                    fault_pc = pc
                    trace.append(SeqTraceEntry(pc, op, fault=Fault.MMIO_EMPTY))
                    break
                if ins.rd != 0:
                    regs[ins.rd] = val
                trace.append(SeqTraceEntry(
                    pc, op,
                    rd=ins.rd if ins.rd else None,
                    value=val if ins.rd else None,
                    mmio=MMIO_READ,
                ))
                pc += 1
                continue
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
                break
            val = int.from_bytes(mem[addr : addr + 4], "little")
            if ins.rd != 0:
                regs[ins.rd] = val
            trace.append(SeqTraceEntry(pc, op, rd=ins.rd if ins.rd else None,
                                       value=val if ins.rd else None))
            pc += 1
        elif op == Op.STORE:
            addr = (regs[ins.rs1] + ins.imm) & MASK32
            if mmio is not None and mmio.contains(addr):
                if not mmio.is_write_reg(addr):
                    fault = Fault.ADDR
                    fault_pc = pc
                    trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
                    break
                data = regs[ins.rs2]
                mmio.write(data)
                trace.append(SeqTraceEntry(
                    pc, op, addr=addr, mem_value=data, mmio=MMIO_WRITE
                ))
                pc += 1
                continue
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
                break
            data = regs[ins.rs2]
            mem[addr : addr + 4] = data.to_bytes(4, "little")
            trace.append(SeqTraceEntry(pc, op, addr=addr, mem_value=data))
            pc += 1
        elif op == Op.BEQ:
            old_pc = pc
            if regs[ins.rs1] == regs[ins.rs2]:
                pc = old_pc + ins.imm
            else:
                pc = old_pc + 1
            trace.append(SeqTraceEntry(old_pc, op))
        else:  # HALT
            halted = True
            trace.append(SeqTraceEntry(pc, op))

    return SeqResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
    )
