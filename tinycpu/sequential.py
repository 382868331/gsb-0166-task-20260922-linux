"""无流水顺序参考解释器（独立实现，不导入 pipeline）。

一条指令完整执行（取指/译码/执行/访存/写回一气呵成）后才执行下一条。
它只与 pipeline 共享 :mod:`tinycpu.isa` 的静态定义（指令格式、校验、
32 位常量）与 :mod:`tinycpu.interrupt` 的控制器/事件类型，期望状态由
这里独立推导，用于交叉验证流水线的退休结果。

精确异常语义对齐：遇到非法 LOAD/STORE 时该指令不产生任何架构效果
（LOAD 不写寄存器、STORE 不写内存），机器停在故障点，此前指令全部提交。

中断语义对齐：中断只在退休边界评估（一条指令完整退休之后）；接纳时
保存下一条指令的 PC、转向固定向量并屏蔽嵌套；IRET 恢复保存的 PC，
其退休边界之后才解除屏蔽；异常优先，HALT 退休后不再接纳。注入用
与流水线相同的退休边界编号，因此两侧事件逐点可比。
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
from .interrupt import InterruptController, IrqEvent, check_interrupt_setup


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


@dataclass(frozen=True, slots=True)
class SeqResult:
    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[SeqTraceEntry, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]
    # 中断：接纳事件（含保存 PC）、运行结束时的 pending/屏蔽状态。
    irq_events: tuple[IrqEvent, ...] = ()
    irq_pending: bool = False
    irq_masked: bool = False


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def run_sequential(
    instructions: list[Instruction] | tuple[Instruction, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
    *,
    interrupts: Optional[InterruptController] = None,
    interrupt_vector: Optional[int] = None,
) -> SeqResult:
    """顺序解释执行直到 HALT 或非法访存。

    与 :func:`tinycpu.pipeline.run` 使用完全相同的静态校验，
    因此两者对同一程序要么都接受、要么都拒绝。中断参数语义
    （控制器、固定向量、退休边界注入）也与流水线完全一致。
    """
    prog, mem, regs = validate_program(instructions, memory, initial_regs)
    n = len(prog)
    vector = check_interrupt_setup(interrupts, interrupt_vector, n)

    trace: list[SeqTraceEntry] = []
    pc = 0
    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    irq_events: list[IrqEvent] = []
    irq_masked = False
    irq_saved_pc: Optional[int] = None
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
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
            else:
                val = int.from_bytes(mem[addr : addr + 4], "little")
                if ins.rd != 0:
                    regs[ins.rd] = val
                trace.append(SeqTraceEntry(pc, op, rd=ins.rd if ins.rd else None,
                                           value=val if ins.rd else None))
                pc += 1
        elif op == Op.STORE:
            addr = (regs[ins.rs1] + ins.imm) & MASK32
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                trace.append(SeqTraceEntry(pc, op, fault=Fault.ADDR))
            else:
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
        elif op == Op.IRET:
            # 中断返回：恢复保存的 PC；无中断上下文时为空操作。
            # 解除屏蔽发生在本退休边界的评估之后（见下方）。
            old_pc = pc
            if irq_masked and irq_saved_pc is not None:
                pc = irq_saved_pc
            else:
                pc += 1
            trace.append(SeqTraceEntry(old_pc, op))
        else:  # HALT
            halted = True
            trace.append(SeqTraceEntry(pc, op))

        # ---- 退休边界：中断脉冲到达 / 接纳 / IRET 解除屏蔽 ----
        # 与流水线同一契约：异常优先（故障边界不接纳）、HALT 后不接纳、
        # IRET 自己的边界仍按屏蔽评估，最早下一个边界才可再次接纳。
        if interrupts is not None:
            boundary = len(trace) - 1
            interrupts._arrive(boundary)
            accepted = False
            if (
                fault is None
                and not halted
                and interrupts.pending
                and not irq_masked
            ):
                irq_saved_pc = pc  # 下一条尚未退休指令的 PC
                irq_events.append(
                    IrqEvent(
                        boundary=boundary,
                        saved_pc=pc,
                        vector_pc=vector,
                        cycle=None,
                    )
                )
                irq_masked = True
                interrupts._take()
                pc = vector
                accepted = True
            if op == Op.IRET and not accepted:
                irq_masked = False

    return SeqResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
        irq_events=tuple(irq_events),
        irq_pending=interrupts.pending if interrupts is not None else False,
        irq_masked=irq_masked,
    )
