"""五级流水线周期内核（IF/ID/EX/MEM/WB），带精确异常语义。

时序约定（每个周期）：
  1. WB  退休/提交：寄存器写、STORE 内存写、HALT、非法访存报告都在这里；
  2. MEM LOAD 读内存（地址非法时不实际访问，数据置 0，等 WB 报告）；
  3. EX  ALU、地址计算、BEQ 决策（命中则冲刷 ID/IF 两条年轻指令）；
  4. ID  读寄存器（本周期 WB 先写后读）、load-use 冒险检测；
  5. IF  取指。

关键性质：
* 所有架构状态变更（寄存器写、内存写、HALT、fault）只在 WB 发生，
  因此 WB 报告非法访存时，同周期 MEM 里更年轻的 STORE 一定还没写内存；
* load-use：EX 里是 LOAD、ID 里的指令要用其 rd 时，冻结 PC 与 IF/ID，
  向 EX 插入一个气泡；
* 前递：EX 取操作数时 MEM 锁存器（最新）优先，其次 WB 锁存器，
  再次 ID 级捕获的寄存器值；
* BEQ 在 EX 决策，命中时把当周期 ID、IF 中的两条年轻指令标记为冲刷。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .isa import (
    MASK32,
    MAX_CYCLES,
    CycleLimitError,
    Fault,
    Instruction,
    Op,
    validate_program,
)

# ---------------------------------------------------------------------------
# 对外数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Slot:
    """某个周期某个流水级的内容。

    index: 指令下标（气泡且无受害者时为 None）。
    bubble: True 表示该级本周期不执行有效指令。
    reason: None 正常；"stall" load-use 气泡；"flush" 分支冲刷；
            "hold" load-use 时被冻结的 IF/ID；"drain" HALT 后排空/启动填充。
    """

    stage: str
    index: Optional[int]
    bubble: bool
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CycleSlots:
    """一个周期的五个流水槽（顺序 IF/ID/EX/MEM/WB）。"""

    cycle: int
    slots: tuple[Slot, Slot, Slot, Slot, Slot]

    @property
    def pc_tuple(self) -> tuple[Optional[int], ...]:
        return tuple(s.index for s in self.slots)


@dataclass(frozen=True, slots=True)
class RetireRecord:
    """退休轨迹中的一条（严格按 WB 退休顺序）。

    架构效果四元组：
    * fault 非 None      —— 该指令触发异常，不写寄存器/内存；
    * rd/value 非 None   —— 提交了一个寄存器写（r0 的写被架构丢弃，不记录）；
    * addr/mem_value     —— 提交了一个 32 位 STORE；
    * HALT/BEQ/普通算术无写时只有 pc/op。
    """

    cycle: int
    pc: int
    op: Op
    fault: Optional[Fault] = None
    rd: Optional[int] = None
    value: Optional[int] = None
    addr: Optional[int] = None
    mem_value: Optional[int] = None

    @property
    def committed(self) -> bool:
        return self.fault is None


@dataclass(frozen=True, slots=True)
class RunResult:
    """一次流水线运行的完整结果。"""

    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[RetireRecord, ...]
    cycles: tuple[CycleSlots, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]
    fault_cycle: Optional[int]

    @property
    def cycles_used(self) -> int:
        return len(self.cycles)


# ---------------------------------------------------------------------------
# 内部锁存器
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Latch:
    valid: bool = False
    pc: int = -1
    insn: Optional[Instruction] = None
    # 无效气泡的可视化原因（"stall"/"flush"）；启动/排空气泡为 None。
    bubble_reason: Optional[str] = None
    # EX 产物
    rs1_val: int = 0
    rs2_val: int = 0
    result: int = 0          # ALU/ADDI 结果
    addr: int = 0            # LOAD/STORE 有效字节地址
    bad_addr: bool = False   # EX 已判定非法，但延迟到 WB 报告
    store_data: int = 0      # STORE 要写的 32 位值
    # MEM 产物
    load_data: int = 0


# ---------------------------------------------------------------------------
# 周期内核
# ---------------------------------------------------------------------------


def run(
    instructions: list[Instruction] | tuple[Instruction, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
) -> RunResult:
    """跑五级流水线直到 HALT 退休、非法访存报告或周期预算耗尽。

    参数静态校验全部在第一条指令执行前完成，失败不会产生任何部分结果。

    Raises:
        TypeError/ValueError/ProgramError: 输入不合法。
        CycleLimitError: 跑满 ``MAX_CYCLES`` 仍未结束（预算耗尽，区别于故障）。
    """
    prog, mem, regs = validate_program(instructions, memory, initial_regs)
    n = len(prog)

    if_id = _Latch()
    id_ex = _Latch()
    ex_mem = _Latch()
    mem_wb = _Latch()

    pc = 0
    trace: list[RetireRecord] = []
    cycle_log: list[CycleSlots] = []

    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    fault_cycle: Optional[int] = None

    for cycle in range(1, MAX_CYCLES + 1):
        # 本周期控制信号
        wb_fault_this_cycle = False
        stall = False
        flush = False
        flush_victim_id: Optional[int] = None
        flush_victim_if: Optional[int] = None

        # ---- 记录本周期各级"占据者"（控制信号可能改写 ID/IF 的标记） ----
        slot_if = Slot("IF", pc if pc < n else None, pc >= n,
                       "drain" if pc >= n else None)
        slot_id = Slot(
            "ID", if_id.pc if if_id.valid else None, bubble=not if_id.valid,
            reason=None if if_id.valid else "drain",
        )
        slot_ex = Slot(
            "EX", id_ex.pc if id_ex.valid else None, bubble=not id_ex.valid,
            reason=None if id_ex.valid else id_ex.bubble_reason or "drain",
        )
        slot_me = Slot(
            "MEM", ex_mem.pc if ex_mem.valid else None, bubble=not ex_mem.valid,
            reason=None if ex_mem.valid else ex_mem.bubble_reason or "drain",
        )
        slot_wb = Slot(
            "WB", mem_wb.pc if mem_wb.valid else None, bubble=not mem_wb.valid,
            reason=None if mem_wb.valid else mem_wb.bubble_reason or "drain",
        )

        # ==============================================================
        # 1) WB：退休。所有架构提交只发生在这里。
        # ==============================================================
        if mem_wb.valid:
            w = mem_wb.insn
            assert w is not None
            if w.op in (Op.LOAD, Op.STORE) and mem_wb.bad_addr:
                # 非法访存到 WB 才报告：本指令不提交，
                # 同周期 MEM/EX/ID/IF 全部更年轻，一律不提交。
                wb_fault_this_cycle = True
                fault = Fault.ADDR
                fault_pc = mem_wb.pc
                fault_cycle = cycle
                trace.append(
                    RetireRecord(cycle=cycle, pc=mem_wb.pc, op=w.op, fault=Fault.ADDR)
                )
                slot_wb = Slot("WB", mem_wb.pc, bubble=False, reason="fault")
            else:
                rec_kwargs: dict = {"cycle": cycle, "pc": mem_wb.pc, "op": w.op}
                if w.op in (Op.ADD, Op.SUB, Op.ADDI):
                    if w.rd != 0:
                        regs[w.rd] = mem_wb.result
                        rec_kwargs.update(rd=w.rd, value=mem_wb.result)
                elif w.op == Op.LOAD:
                    if w.rd != 0:
                        regs[w.rd] = mem_wb.load_data
                        rec_kwargs.update(rd=w.rd, value=mem_wb.load_data)
                elif w.op == Op.STORE:
                    a = mem_wb.addr
                    mem[a : a + 4] = mem_wb.store_data.to_bytes(4, "little")
                    rec_kwargs.update(addr=a, mem_value=mem_wb.store_data)
                elif w.op == Op.HALT:
                    halted = True
                trace.append(RetireRecord(**rec_kwargs))

        # ==============================================================
        # 2) MEM：LOAD 读内存（非法地址不触碰内存）；STORE 等到 WB 才写。
        #    （wb_fault 时 MEM 的年轻指令即使读了也不会被前递到任何会提交
        #      的指令，锁存器整体丢弃即可。）
        # ==============================================================
        if ex_mem.valid and not wb_fault_this_cycle:
            m = ex_mem.insn
            assert m is not None
            if m.op == Op.LOAD and not ex_mem.bad_addr:
                a = ex_mem.addr
                ex_mem.load_data = int.from_bytes(mem[a : a + 4], "little")

        # ==============================================================
        # 3) EX：ALU / 地址 / 分支决策（含前递）。
        # ==============================================================
        if id_ex.valid and not wb_fault_this_cycle:
            x = id_ex.insn
            assert x is not None

            v1 = _forward_operand(1, x, id_ex, ex_mem, mem_wb, regs)
            v2 = _forward_operand(2, x, id_ex, ex_mem, mem_wb, regs)

            if x.op == Op.ADD:
                id_ex.result = (v1 + v2) & MASK32
            elif x.op == Op.SUB:
                id_ex.result = (v1 - v2) & MASK32
            elif x.op == Op.ADDI:
                id_ex.result = (v1 + x.imm) & MASK32
            elif x.op in (Op.LOAD, Op.STORE):
                addr = (v1 + x.imm) & MASK32
                id_ex.addr = addr
                id_ex.bad_addr = not _addr_ok(addr, len(mem))
                if x.op == Op.STORE:
                    id_ex.store_data = v2
            elif x.op == Op.BEQ:
                if v1 == v2:
                    flush = True
                    flush_victim_id = if_id.pc if if_id.valid else None
                    flush_victim_if = pc if pc < n else None
                    pc = id_ex.pc + x.imm  # 已静态校验落在 [0, n)
                    slot_ex = Slot("EX", id_ex.pc, bubble=False, reason="take")
            # HALT：EX 无动作

        # ==============================================================
        # 4) ID：load-use 冒险检测（WB 已故障则整管线下沉丢弃）。
        # ==============================================================
        if (
            if_id.valid
            and id_ex.valid
            and not wb_fault_this_cycle
            and not flush
        ):
            d = if_id.insn
            e = id_ex.insn
            assert d is not None and e is not None
            if e.op == Op.LOAD and e.rd != 0 and _uses(d, e.rd):
                stall = True

        # ==============================================================
        # 5) 流水槽标记（冲刷/停顿可视化）。
        # ==============================================================
        if wb_fault_this_cycle:
            # 年轻各级标记为丢弃。
            if ex_mem.valid:
                slot_me = Slot("MEM", ex_mem.pc, bubble=True, reason="fault")
            if id_ex.valid:
                slot_ex = Slot("EX", id_ex.pc, bubble=True, reason="fault")
            if if_id.valid:
                slot_id = Slot("ID", if_id.pc, bubble=True, reason="fault")
            if pc < n:
                slot_if = Slot("IF", pc, bubble=True, reason="fault")
        elif flush:
            if flush_victim_id is not None:
                slot_id = Slot("ID", flush_victim_id, bubble=True, reason="flush")
            if flush_victim_if is not None:
                slot_if = Slot("IF", flush_victim_if, bubble=True, reason="flush")
        elif stall:
            # 停顿周期：LOAD 正在 EX 执行（保持其有效槽标记），
            # ID/IF 冻结；stall 气泡要到下一周期才出现在 EX（见下沉）。
            slot_id = Slot("ID", if_id.pc, bubble=False, reason="hold")
            slot_if = Slot("IF", pc, bubble=False, reason="hold")

        cycle_log.append(
            CycleSlots(cycle, (slot_if, slot_id, slot_ex, slot_me, slot_wb))
        )

        # 异常：本周期结束即停（WB 老指令已提交，年轻指令全部未落架构状态）。
        if wb_fault_this_cycle:
            break
        if halted:
            break

        # ==============================================================
        # 锁存器下沉（组合逻辑结果在本周期末打入）。
        # ==============================================================
        # WB <- MEM，MEM <- EX（整条锁存器复制）。
        mem_wb = ex_mem
        ex_mem = id_ex
        # EX <- ID
        if stall:
            # LOAD 已正常进入 MEM；此处插入 stall 气泡（下一周期在 EX 可见），
            # IF/ID 保持冻结。
            id_ex = _Latch(bubble_reason="stall")
        elif flush:
            # 冲刷气泡下一周期在 EX 可见；IF/ID 为普通排空槽，
            # 分支目标在新 PC 下重新取指。
            id_ex = _Latch(bubble_reason="flush")
        else:
            id_ex = _new_id_ex(if_id, regs)

        # ID <- IF（停顿保持；冲刷插普通气泡；否则取当前 PC 指令）
        if stall:
            pass  # if_id 保持
        elif flush:
            if_id = _Latch()
        else:
            if pc < n:
                if_id = _Latch(valid=True, pc=pc, insn=prog[pc])
            else:
                if_id = _Latch()

        # PC（停顿冻结；冲刷已在 EX 改写；否则 +1）
        if not stall and not flush:
            pc += 1

    else:
        # for/else：预算耗尽，且最后一个周期也没有 halt/fault。
        raise CycleLimitError(
            f"程序在 {MAX_CYCLES} 个周期内未结束（可能存在 BEQ 死循环）"
        )

    return RunResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        cycles=tuple(cycle_log),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
        fault_cycle=fault_cycle,
    )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def _uses(d: Instruction, reg: int) -> bool:
    """ID 级指令 d 是否在 EX 读取寄存器 reg。"""
    if d.op in (Op.ADD, Op.SUB, Op.BEQ, Op.STORE):
        return d.rs1 == reg or d.rs2 == reg
    if d.op in (Op.ADDI, Op.LOAD):
        return d.rs1 == reg
    return False


def _forward_operand(
    which: int,
    x: Instruction,
    id_ex: _Latch,
    ex_mem: _Latch,
    mem_wb: _Latch,
    regs: list[int],
) -> int:
    """EX 取第 which(1/2) 个操作数，MEM 锁存器（最新）优先。"""
    r = x.rs1 if which == 1 else x.rs2
    captured = id_ex.rs1_val if which == 1 else id_ex.rs2_val
    if r != 0 and ex_mem.valid and ex_mem.insn is not None:
        p = ex_mem.insn
        if p.writes_rd and p.rd == r:
            return ex_mem.load_data if p.op == Op.LOAD else ex_mem.result
    if r != 0 and mem_wb.valid and mem_wb.insn is not None:
        q = mem_wb.insn
        if q.writes_rd and q.rd == r:
            return mem_wb.load_data if q.op == Op.LOAD else mem_wb.result
    return captured


def _new_id_ex(if_id: _Latch, regs: list[int]) -> _Latch:
    """IF/ID 锁存器下沉到 ID/EX，同时在 ID 级捕获寄存器值。"""
    if not if_id.valid or if_id.insn is None:
        return _Latch()
    d = if_id.insn
    return _Latch(
        valid=True,
        pc=if_id.pc,
        insn=d,
        rs1_val=regs[d.rs1],
        rs2_val=regs[d.rs2],
    )
