"""无流水顺序解释器——独立参考实现。

本模块**不导入也不调用** ``pipeline_cpu.core``：它按“取指—执行—
提交”一次一条的顺序模型独立计算期望值，只共享 ``model`` 中的数据
类型与输入校验（数据类型定义不属于被测核心算法）。

状态快照口径
------------
顺序模型在即将执行故障指令*之前*停止，因此其寄存器与内存正是
“异常前”状态；流水线在故障退休的同一周期冻结所有年轻阶段，二者
逐值可比：

* 故障指令及更年轻指令均未产生任何结构效果；
* 更老指令全部按序提交（含早于故障的 STORE 已写内存）。
"""

from __future__ import annotations

from .model import (
    Fault,
    Instruction,
    Opcode,
    ReferenceResult,
    WORD_MASK,
    validate_program,
    _check_int,
    MAX_CYCLES,
)


def _mask32(x: int) -> int:
    return x & WORD_MASK


def run_sequential(program: object, memory: object,
                   step_budget: int = MAX_CYCLES) -> ReferenceResult:
    """顺序执行程序直到 HALT、故障或步数预算耗尽。

    ``step_budget`` 仅为防御性上限（默认 2000）；本 ISA 的控制流
    静态固定，正常程序必然终止或落出程序区触发 bad_fetch。
    耗尽时抛 :class:`~pipeline_cpu.model.CycleLimitExceeded`。
    """
    # 延迟导入以保持“不依赖 core”；这里只复用错误类型。
    from .model import InvalidProgramError, CycleLimitExceeded

    sb = _check_int(step_budget, "step_budget")
    if not 1 <= sb <= MAX_CYCLES:
        raise InvalidProgramError(
            f"step_budget 必须在 1..{MAX_CYCLES} 内，得到 {sb}")

    prog: tuple[Instruction, ...]
    mem: bytearray
    prog, mem = validate_program(program, memory)
    n = len(prog)
    regs = [0] * 32
    pc = 0
    executed: list[tuple[int, int | None, int | None, bool]] = []

    for _step in range(step_budget + 1):
        # ---- 取指 ----
        if not (pc % 4 == 0 and 0 <= pc < 4 * n):
            return ReferenceResult(
                halted=False,
                fault=Fault("bad_fetch", _mask32(pc)),
                fault_seq=None,
                executed=tuple(executed),
                final_registers=tuple(regs),
                final_memory=bytes(mem),
                executed_count=len(executed),
            )
        seq = pc // 4
        inst = prog[seq]
        op = inst.opcode

        # ---- 执行 + 提交（故障指令在产生任何效果前中止） ----
        next_pc = _mask32(pc + 4)
        if op in (Opcode.ADD, Opcode.SUB):
            x = regs[inst.b]
            y = regs[inst.c]
            val = _mask32(x + y if op is Opcode.ADD else x - y)
            rd, val_out = inst.a, (None if inst.a == 0 else val)
            if inst.a != 0:
                regs[inst.a] = val
            executed.append((seq, rd, val_out, False))
        elif op is Opcode.ADDI:
            val = _mask32(regs[inst.b] + inst.imm)
            rd, val_out = inst.a, (None if inst.a == 0 else val)
            if inst.a != 0:
                regs[inst.a] = val
            executed.append((seq, rd, val_out, False))
        elif op is Opcode.LOAD:
            addr = _mask32(regs[inst.b] + inst.imm)
            if not (addr % 4 == 0 and 0 <= addr <= len(mem) - 4):
                return ReferenceResult(
                    halted=False, fault=Fault("bad_address", addr),
                    fault_seq=seq, executed=tuple(executed),
                    final_registers=tuple(regs), final_memory=bytes(mem),
                    executed_count=len(executed))
            val = int.from_bytes(mem[addr:addr + 4], "little")
            rd, val_out = inst.a, (None if inst.a == 0 else val)
            if inst.a != 0:
                regs[inst.a] = val
            executed.append((seq, rd, val_out, False))
        elif op is Opcode.STORE:
            addr = _mask32(regs[inst.b] + inst.imm)
            if not (addr % 4 == 0 and 0 <= addr <= len(mem) - 4):
                # 关键：先检查后写，故障前内存不含本次写入。
                return ReferenceResult(
                    halted=False, fault=Fault("bad_address", addr),
                    fault_seq=seq, executed=tuple(executed),
                    final_registers=tuple(regs), final_memory=bytes(mem),
                    executed_count=len(executed))
            mem[addr:addr + 4] = _mask32(regs[inst.c]).to_bytes(4, "little")
            executed.append((seq, None, None, False))
        elif op is Opcode.BEQ:
            if regs[inst.b] == regs[inst.c]:
                next_pc = _mask32(pc + inst.imm)
            executed.append((seq, None, None, False))
        elif op is Opcode.HALT:
            executed.append((seq, None, None, True))
            return ReferenceResult(
                halted=True, fault=None, fault_seq=None,
                executed=tuple(executed),
                final_registers=tuple(regs), final_memory=bytes(mem),
                executed_count=len(executed))
        pc = next_pc

    raise CycleLimitExceeded(f"顺序执行超过 {step_budget} 步预算")
