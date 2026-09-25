"""无流水顺序参考解释器（独立实现，不导入 pipeline）。

一条指令完整执行（取指/译码/执行/访存/写回一气呵成）后才执行下一条。
它只与 pipeline 共享 :mod:`tinycpu.isa` 的静态定义（指令格式、校验、
32 位常量），期望状态由这里独立推导，用于交叉验证流水线的退休结果。

精确异常语义对齐：遇到非法 LOAD/STORE 时该指令不产生任何架构效果
（LOAD 不写寄存器、STORE 不写内存），机器停在故障点，此前指令全部提交。

外部中断语义同样在这里**独立**实现（不抄 pipeline 的锁存器逻辑）：
脉冲登记在"退休边界"上（:class:`tinycpu.interrupt.BoundaryInterrupts`，
k = 此前已退休指令数）；一条指令正常执行完、进入下一条之前采样：

* 同一条非法访存指令的边界上若有脉冲：先记异常停机，中断仍 pending
  （异常优先），即脉冲不被消费；
* HALT 退休后不接纳；IRET 自己的边界不接纳；
* 接纳时保存"下一条顺序指令"PC（顺序机没有在途指令，故就是下一条
  将要执行的 pc），转固定向量 VECTOR_PC，置屏蔽；
* 屏蔽期脉冲只置一位 pending；IRET 执行后恢复保存 PC、冲刷在序概念
  （顺序机本就没有在途指令）并解除屏蔽，pending 留给恢复后第一条
  指令的退休边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .interrupt import BoundaryInterrupts, InterruptAccept
from .isa import (
    MASK32,
    VECTOR_PC,
    Fault,
    Instruction,
    InterruptStateError,
    Op,
    validate_program,
)


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
    event: Optional[str] = None
    saved_pc: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SeqResult:
    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[SeqTraceEntry, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]
    interrupts: tuple[InterruptAccept, ...] = ()
    interrupts_pending: bool = False
    masked: bool = False


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def run_sequential(
    instructions: list[Instruction] | tuple[Instruction, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
    interrupts: Optional[BoundaryInterrupts] = None,
) -> SeqResult:
    """顺序解释执行直到 HALT 或非法访存。

    与 :func:`tinycpu.pipeline.run` 使用完全相同的静态校验，
    因此两者对同一程序要么都接受、要么都拒绝。
    ``interrupts`` 为退休边界脉冲表；``None`` 表示无中断。
    """
    if interrupts is not None and not isinstance(interrupts, BoundaryInterrupts):
        raise TypeError(
            "interrupts 必须是 tinycpu.BoundaryInterrupts 或 None，"
            f"收到 {type(interrupts).__name__}"
        )
    prog, mem, regs = validate_program(instructions, memory, initial_regs)

    trace: list[SeqTraceEntry] = []
    pc = 0
    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    interrupt_log: list[InterruptAccept] = []

    # 中断状态（与 pipeline 同名同义，但由顺序逻辑独立维护）。
    pending = False
    masked = False
    saved_pc: Optional[int] = None

    # 顺序机不会有周期预算问题，但防御失控分支：程序最多走 10_000 步。
    steps = 0
    max_steps = 10_000

    while not halted and fault is None:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("顺序参考执行步数超限（程序可能含无限分支循环）")
        ins = prog[pc]
        op = ins.op

        # 当边界脉冲：以"已退休指令数"编号。IRET 也先并入 pending，
        # 因为 IRET 自己的退休边界不接纳。
        retired_count = len(trace)
        if interrupts is not None and interrupts.has_pulse(retired_count):
            pending = True

        rec_kwargs: dict = {}
        next_pc = pc + 1
        fault_this_insn = False

        if op == Op.ADD:
            val = (regs[ins.rs1] + regs[ins.rs2]) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            rec_kwargs.update(rd=ins.rd if ins.rd else None,
                              value=val if ins.rd else None)
        elif op == Op.SUB:
            val = (regs[ins.rs1] - regs[ins.rs2]) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            rec_kwargs.update(rd=ins.rd if ins.rd else None,
                              value=val if ins.rd else None)
        elif op == Op.ADDI:
            val = (regs[ins.rs1] + ins.imm) & MASK32
            if ins.rd != 0:
                regs[ins.rd] = val
            rec_kwargs.update(rd=ins.rd if ins.rd else None,
                              value=val if ins.rd else None)
        elif op == Op.LOAD:
            addr = (regs[ins.rs1] + ins.imm) & MASK32
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                fault_this_insn = True
                rec_kwargs.update(fault=Fault.ADDR)
            else:
                val = int.from_bytes(mem[addr : addr + 4], "little")
                if ins.rd != 0:
                    regs[ins.rd] = val
                rec_kwargs.update(rd=ins.rd if ins.rd else None,
                                  value=val if ins.rd else None)
        elif op == Op.STORE:
            addr = (regs[ins.rs1] + ins.imm) & MASK32
            if not _addr_ok(addr, len(mem)):
                fault = Fault.ADDR
                fault_pc = pc
                fault_this_insn = True
                rec_kwargs.update(fault=Fault.ADDR)
            else:
                data = regs[ins.rs2]
                mem[addr : addr + 4] = data.to_bytes(4, "little")
                rec_kwargs.update(addr=addr, mem_value=data)
        elif op == Op.BEQ:
            if regs[ins.rs1] == regs[ins.rs2]:
                next_pc = pc + ins.imm
        elif op == Op.IRET:
            if not masked or saved_pc is None:
                raise InterruptStateError(
                    f"IRET 在 pc={pc} 执行时没有活跃中断（未屏蔽、无保存 PC）"
                )
            rec_kwargs.update(event="IRET", saved_pc=saved_pc)
        else:  # HALT
            pass

        trace.append(SeqTraceEntry(pc, op, **rec_kwargs))

        # ---- 退休后的边界决策 ----
        if fault_this_insn:
            # 异常优先：停机，当拍脉冲即使刚并入也保持 pending（不消费）。
            break

        if op == Op.HALT:
            # HALT 退休后不再接纳中断。
            halted = True
            break

        if op == Op.IRET:
            # 恢复保存 PC 后才解除屏蔽；本边界不接纳（pending 保留到
            # 恢复流第一条指令的退休边界）。
            pc = saved_pc
            masked = False
            saved_pc = None
            continue

        if pending and not masked:
            # 接纳：保存"下一条尚未退休指令"——顺序机即下一条将要执行的
            # 指令（分支时是分支目标），转固定向量，冲刷（丢弃 next_pc
            # 概念上的在途取指），屏蔽嵌套。
            pending = False
            masked = True
            interrupt_log.append(
                InterruptAccept(
                    cycle=retired_count, saved_pc=next_pc, retired_pc=pc
                )
            )
            # 在刚写入的轨迹记录上补注接纳事件（不可变对象 -> 替换）。
            trace[-1] = SeqTraceEntry(
                pc, op, event="INTR", saved_pc=next_pc,
                **{
                    k: v
                    for k, v in rec_kwargs.items()
                    if k not in ("event", "saved_pc")
                },
            )
            saved_pc = next_pc
            pc = VECTOR_PC
            continue

        pc = next_pc

    return SeqResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
        interrupts=tuple(interrupt_log),
        interrupts_pending=pending,
        masked=masked,
    )
