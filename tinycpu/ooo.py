"""带 4 项重排序窗口（ROB）与寄存器重命名的乱序执行核心。

与 :mod:`tinycpu.pipeline` 的五级顺序流水线**并存**、互不依赖：

* 派发（dispatch/rename）按程序序进入固定 4 项的 ROB；
* 整数算术（ADD/SUB/ADDI）与 BEQ 只要操作数就绪即可**乱序完成**，
  取值通过重命名标签从已完成项前递，不读未提交的架构寄存器；
* 退休严格按程序序：寄存器写、STORE 内存写、HALT、访存故障报告都只在
  ROB 头部退休时发生——精确异常由按序退休自然保证；
* BEQ 静态预测"不命中"。命中即误预测：回收该分支之后所有年轻 ROB 项
  （包括已经完成的算术结果），恢复重命名映射，PC 改向目标重新取指；
* 访存严格保序：LOAD 只有在所有更老的 LOAD/STORE 都已退休、且更老的
  BEQ 都已解析后才会启动；STORE 仅在退休时写内存。因此 LOAD/STORE
  永不在未解析分支之后投机执行（本迭代不做推测访存）。

时序模型（每周期固定阶段顺序）：
  A. 退休   ：ROB 头一项若已完成则提交（含 STORE 写内存 / HALT / 故障）；
  B. 完成   ：访存中的 LOAD 走节拍，到时数据就绪（LOAD_LATENCY 拍）；
  C. 派发   ：ROB 有空位且前端未停则取入一条并做寄存器重命名，否则记停顿；
  D. 执行   ：从老到轻扫描，选第一个可执行的忙项（最老就绪策略，每周期 1 条）；
  E. 回收   ：若本周期 BEQ 解析为命中，回收年轻项、恢复映射、PC 改向。

ALU/BEQ/STORE 在选中当周期即完成；LOAD 选中后经 LOAD_LATENCY 拍就绪。
HALT 在派发时即标记完成（不占执行选择）。期望值不来自本模块：测试用
:mod:`tinycpu.sequential` 的独立顺序解释器与手算例子核对。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .isa import (
    MASK32,
    MAX_CYCLES,
    CycleLimitError,
    Fault,
    Op,
    validate_program,
)

# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

ROB_SIZE = 4
"""重排序窗口项数（题面指定 4 项）。"""

LOAD_LATENCY = 3
"""LOAD 从被执行选择到数据就绪的周期数（地址生成 1 拍 + 访存 2 拍）。"""

# ROB 项状态：
_BUSY = "busy"   # 已派发，尚未开始执行
_MEM = "mem"     # LOAD 已算地址，正在访存节拍中
_DONE = "done"   # 结果就绪 / STORE 地址与数据就绪 / BEQ 已解析 / HALT


# ---------------------------------------------------------------------------
# 对外事件数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchEvent:
    """一条指令的派发（译码+重命名+入 ROB）事件。

    pc:        静态指令下标（动态重复执行时同一 pc 会多次出现）。
    seq:       全局单调递增的动态标签（也是重命名目标标签）。
    rd:        被重命名的架构寄存器号；r0/无写回指令为 None。
    old_tag:   派发前 rd 对应的旧标签（用于误预测恢复），无则 None。
    src:       两个源操作数的标签快照（None 表示读已提交的架构寄存器）。
    occupancy: 派发后 ROB 内项数（1..ROB_SIZE）。
    """

    pc: int
    seq: int
    rd: Optional[int]
    old_tag: Optional[int]
    src: tuple[Optional[int], Optional[int]]
    occupancy: int


@dataclass(frozen=True, slots=True)
class SquashEvent:
    """BEQ 误预测（实际命中，预测为不命中）后的年轻结果回收。"""

    cycle: int
    branch_pc: int
    target: int
    squashed: tuple[int, ...]
    """被回收的年轻指令静态下标（按程序序）。"""
    reclaimed_completed: tuple[int, ...]
    """其中已经完成、结果被丢弃的下标（整数算术乱序完成的直接证据）。"""


@dataclass(frozen=True, slots=True)
class OOORetireRecord:
    """退休轨迹中的一条（严格按程序序，字段与顺序参考逐条对齐）。

    按序退休下，每条退休的写回就是当时最新的已提交写：即使更年轻的同
    寄存器指令已夺走 RAT 映射，这里仍无条件更新架构寄存器——年轻指令若
    之后被回收或自身故障不写回，架构状态就停在本条，精确状态由此保持。
    """

    cycle: int
    pc: int
    seq: int
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
class OOOCycle:
    """一个周期内五个阶段的事件快照。

    dispatch:       本周期派发事件（无派发为 None）。
    execute/complete: 本周期开始执行 / 结果就绪的指令静态下标（按程序序）。
                      ALU、STORE、BEQ 同一周期同时出现在两者中；
                      LOAD 先在 execute、若干周期后才在 complete。
    retire:         本周期退休的指令下标（无则 None）。
    squash:         误预测回收事件（无则 None）。
    dispatch_stall: 前端本周期未能派发的原因："rob_full"；否则 None。
    fault_discard:  头部指令报告 ADDR 故障时，被一并丢弃的年轻下标。
    """

    cycle: int
    dispatch: Optional[DispatchEvent] = None
    execute: tuple[int, ...] = ()
    complete: tuple[int, ...] = ()
    retire: Optional[int] = None
    squash: Optional[SquashEvent] = None
    dispatch_stall: Optional[str] = None
    fault_discard: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OOORunResult:
    """一次乱序核心运行的完整结果。"""

    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[OOORetireRecord, ...]
    cycles: tuple[OOOCycle, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]
    fault_cycle: Optional[int]
    squashes: tuple[SquashEvent, ...]
    """运行中全部误预测回收事件（按周期）。"""

    @property
    def cycles_used(self) -> int:
        return len(self.cycles)


# ---------------------------------------------------------------------------
# 内部 ROB 项
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Entry:
    seq: int
    pc: int
    insn: object
    src1: Optional[int]      # 源标签；None = 读架构寄存器
    src2: Optional[int]
    rd: Optional[int]        # 架构目标寄存器；None = 不重命名（r0/无写）
    old_tag: Optional[int]   # rd 派发前的旧标签
    state: str = _BUSY
    result: int = 0          # ALU/LOAD 结果
    addr: int = 0
    bad_addr: bool = False
    store_data: int = 0
    taken: bool = False      # BEQ 解析结果
    mem_left: int = 0        # LOAD 剩余访存节拍


# ---------------------------------------------------------------------------
# 周期内核
# ---------------------------------------------------------------------------


def run_ooo(
    instructions: list[object] | tuple[object, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
) -> OOORunResult:
    """在 4 项 ROB + 寄存器重命名的乱序核心上执行程序。

    参数与静态校验与 :func:`tinycpu.pipeline.run` 完全一致
    （复用 ``isa.validate_program``）：所有非法输入在第一个周期之前拒绝，
    校验在内存副本上进行，调用方传入的 ``bytearray`` 不会被改动。

    Returns:
        :class:`OOORunResult`。

    Raises:
        TypeError/ValueError/ProgramError: 输入不合法。
        CycleLimitError: 跑满 ``MAX_CYCLES`` 仍未 HALT（预算耗尽）。
    """
    prog, mem, regs = validate_program(instructions, memory, initial_regs)
    n = len(prog)

    # 按程序序排列的在窗 ROB 项；标签 seq 全局单调（退休的标签结果仍保留在
    # prf 中供在窗读者取值，运行期标签数受周期预算上界约束）。
    window: list[_Entry] = []
    prf: dict[int, int] = {}
    rat: list[Optional[int]] = [None] * 32

    pc = 0
    next_seq = 0
    front_alive = True
    trace: list[OOORetireRecord] = []
    cycle_log: list[OOOCycle] = []
    squashes: list[SquashEvent] = []

    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    fault_cycle: Optional[int] = None

    for cycle in range(1, MAX_CYCLES + 1):
        disp_evt: Optional[DispatchEvent] = None
        executed: list[int] = []
        completed: list[int] = []
        retired_pc: Optional[int] = None
        squash_evt: Optional[SquashEvent] = None
        stall_reason: Optional[str] = None
        fault_discard: tuple[int, ...] = ()
        pending: Optional[_Entry] = None
        stop = False

        # ==========================================================
        # A) 退休：ROB 头部完成项按程序序提交（每周期 1 条）。
        # ==========================================================
        if window and window[0].state == _DONE:
            head = window[0]
            w = head.insn

            if w.op in (Op.LOAD, Op.STORE) and head.bad_addr:
                fault = Fault.ADDR
                fault_pc = head.pc
                fault_cycle = cycle
                fault_discard = tuple(e.pc for e in window[1:])
                trace.append(
                    OOORetireRecord(
                        cycle=cycle, pc=head.pc, seq=head.seq,
                        op=w.op, fault=Fault.ADDR,
                    )
                )
                stop = True
            else:
                kw: dict = {
                    "cycle": cycle, "pc": head.pc, "seq": head.seq, "op": w.op,
                }
                if w.op in (Op.ADD, Op.SUB, Op.ADDI):
                    if w.rd != 0:
                        regs[w.rd] = head.result
                        kw.update(rd=w.rd, value=head.result)
                elif w.op == Op.LOAD:
                    if w.rd != 0:
                        regs[w.rd] = head.result
                        kw.update(rd=w.rd, value=head.result)
                elif w.op == Op.STORE:
                    a = head.addr
                    mem[a : a + 4] = head.store_data.to_bytes(4, "little")
                    kw.update(addr=a, mem_value=head.store_data)
                elif w.op == Op.HALT:
                    halted = True
                    stop = True
                trace.append(OOORetireRecord(**kw))

            retired_pc = head.pc
            window.pop(0)

        if stop:
            # HALT 退休或头部报告故障：机器当周期结束，
            # 窗口内剩余年轻项一律不派发/不执行/不提交。
            cycle_log.append(
                OOOCycle(
                    cycle=cycle,
                    retire=retired_pc,
                    fault_discard=fault_discard,
                )
            )
            break

        # ==========================================================
        # B) 完成：访存中的 LOAD 走节拍，到时读内存（非法地址不触碰）。
        #    （严格保序下同一时刻至多一条 LOAD 在访存，仍逐项处理。）
        # ==========================================================
        for e in window:
            if e.state == _MEM:
                e.mem_left -= 1
                if e.mem_left <= 0:
                    e.state = _DONE
                    if not e.bad_addr:
                        a = e.addr
                        e.result = int.from_bytes(mem[a : a + 4], "little")
                    # 结果进入标签物理值（故障 LOAD 的 0 也照发：其年轻
                    # 消费者即便已完成，也会在故障退休时被整体丢弃）。
                    prf[e.seq] = e.result
                    completed.append(e.pc)

        # ==========================================================
        # C) 派发 + 重命名（前端每周期至多 1 条；ROB 满则停顿）。
        # ==========================================================
        if front_alive:
            if len(window) < ROB_SIZE:
                ins = prog[pc]
                seq = next_seq
                next_seq += 1

                def tag_of(reg: int) -> Optional[int]:
                    return None if reg == 0 else rat[reg]

                if ins.op in (Op.ADD, Op.SUB, Op.BEQ):
                    s1, s2 = tag_of(ins.rs1), tag_of(ins.rs2)
                elif ins.op in (Op.ADDI, Op.LOAD):
                    s1, s2 = tag_of(ins.rs1), None
                else:  # STORE
                    s1, s2 = tag_of(ins.rs1), tag_of(ins.rs2)

                rd = ins.rd if ins.writes_rd and ins.rd != 0 else None
                old_tag = rat[rd] if rd is not None else None
                if rd is not None:
                    rat[rd] = seq

                entry = _Entry(
                    seq=seq, pc=pc, insn=ins,
                    src1=s1, src2=s2, rd=rd, old_tag=old_tag,
                )
                if ins.op == Op.HALT:
                    entry.state = _DONE
                    front_alive = False
                window.append(entry)
                disp_evt = DispatchEvent(
                    pc=pc, seq=seq, rd=rd, old_tag=old_tag,
                    src=(s1, s2), occupancy=len(window),
                )
                pc += 1
            else:
                stall_reason = "rob_full"

        # ==========================================================
        # D) 执行：最老就绪策略，每周期选 1 条。
        #    整数算术/分支可乱序；访存额外要求保序与分支已解析。
        # ==========================================================
        for e in window:
            if e.state != _BUSY:
                continue
            if not _operands_ready(e, prf):
                continue
            op = e.insn.op
            if op in (Op.LOAD, Op.STORE) and not _mem_gate_ok(e, window):
                continue

            v1 = _read(e.insn.rs1, e.src1, regs, prf)
            v2 = _read(e.insn.rs2, e.src2, regs, prf)
            executed.append(e.pc)

            if op == Op.ADD:
                e.result = (v1 + v2) & MASK32
                e.state = _DONE
                prf[e.seq] = e.result
                completed.append(e.pc)
            elif op == Op.SUB:
                e.result = (v1 - v2) & MASK32
                e.state = _DONE
                prf[e.seq] = e.result
                completed.append(e.pc)
            elif op == Op.ADDI:
                e.result = (v1 + e.insn.imm) & MASK32
                e.state = _DONE
                prf[e.seq] = e.result
                completed.append(e.pc)
            elif op == Op.LOAD:
                addr = (v1 + e.insn.imm) & MASK32
                e.addr = addr
                e.bad_addr = not _addr_ok(addr, len(mem))
                e.mem_left = LOAD_LATENCY - 1
                e.state = _MEM
            elif op == Op.STORE:
                addr = (v1 + e.insn.imm) & MASK32
                e.addr = addr
                e.bad_addr = not _addr_ok(addr, len(mem))
                e.store_data = v2
                e.state = _DONE
            else:  # BEQ：执行即解析，当周期完成；命中则进入误预测回收。
                e.taken = v1 == v2
                e.state = _DONE
                completed.append(e.pc)
                if e.taken:
                    pending = e
            break  # 每周期只启动一条

        # ==========================================================
        # E) 误预测回收：实际命中（静态预测不命中）。
        #    丢弃分支之后的所有年轻项（含已完成结果），恢复 RAT。
        # ==========================================================
        if pending is not None:
            young = [e for e in window if e.seq > pending.seq]
            reclaimed = [e.pc for e in young if e.state == _DONE]
            # 由轻到老恢复映射：每项被回收时它仍是 rd 的当前属主。
            for e in reversed(young):
                if e.rd is not None:
                    rat[e.rd] = e.old_tag
                prf.pop(e.seq, None)
            window[:] = [e for e in window if e.seq <= pending.seq]
            target = pending.pc + pending.insn.imm  # 静态校验保证在程序内
            squash_evt = SquashEvent(
                cycle=cycle,
                branch_pc=pending.pc,
                target=target,
                squashed=tuple(e.pc for e in young),
                reclaimed_completed=tuple(reclaimed),
            )
            squashes.append(squash_evt)
            pc = target
            front_alive = True

        cycle_log.append(
            OOOCycle(
                cycle=cycle,
                dispatch=disp_evt,
                execute=tuple(executed),
                complete=tuple(completed),
                retire=retired_pc,
                squash=squash_evt,
                dispatch_stall=stall_reason,
            )
        )

        if halted or fault is not None:
            break

    else:
        raise CycleLimitError(
            f"乱序核心在 {MAX_CYCLES} 个周期内未结束（可能存在 BEQ 死循环）"
        )

    return OOORunResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        cycles=tuple(cycle_log),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
        fault_cycle=fault_cycle,
        squashes=tuple(squashes),
    )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def _operands_ready(e: _Entry, prf: dict[int, int]) -> bool:
    return (e.src1 is None or e.src1 in prf) and (
        e.src2 is None or e.src2 in prf
    )


def _mem_gate_ok(e: _Entry, window: list[_Entry]) -> bool:
    """访存保序 + 不做推测访存。

    LOAD/STORE 启动要求：所有更老的 BEQ 已解析，且窗口内没有更老的
    LOAD/STORE（它们只能经退休离窗，故必已按序完成访存/写内存）。
    """
    for older in window:
        if older.seq >= e.seq:
            break
        if older.insn.op == Op.BEQ and older.state != _DONE:
            return False
        if older.insn.op in (Op.LOAD, Op.STORE):
            return False
    return True


def _read(
    reg: int,
    tag: Optional[int],
    regs: list[int],
    prf: dict[int, int],
) -> int:
    if reg == 0:
        return 0
    if tag is None:
        return regs[reg]
    return prf[tag]
