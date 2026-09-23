"""带 4 项重排序窗口与寄存器重命名的乱序执行核心（教学模型）。

与 :mod:`tinycpu.pipeline` 的五级*顺序*流水线并列，互不影响；
``tinycpu.run`` 的旧入口与结果结构保持不变。

模型结构
========

* **重排序窗口（ROB）**：默认 4 项（``window`` 可改，合法范围 1..16）。
  指令按程序序读入窗口，窗口满则取指停顿；**退休严格按程序序**，每周期
  至多 1 条。
* **寄存器重命名**：译码时用 RAT（寄存器别名表）把架构寄存器映射到
  "产生该值的窗口条目标签"；目的寄存器记录旧映射，分支冲刷时按最年轻到
  最老的顺序还原 RAT。标签即窗口条目的单调编号，存活标签数不超过窗口大小。
* **只有整数算术（ADD/SUB/ADDI）可以乱序完成**：操作数就绪即可发射，
  单个整数运算口（每周期至多 1 条，选择"最老的就绪指令"）；其结果仅供
  窗口内通过标签读取，架构寄存器只在退休时写入。
* **访存严格保持顺序**：LOAD/STORE 只有在成为窗口队首（所有更老动态指令
  已退休）且操作数就绪时才执行；STORE 的内存写发生在**退休**时刻。
  因此故障 STORE 之后更年轻的 STORE 物理上不可能已经写内存。
* **BEQ 在操作数就绪后按序解析**（更老的未解析分支优先）。本模型不做
  分支预测器，取指沿"不命中（顺序下一条）"路径进行；解析为命中时，
  窗口中所有更年轻的动态指令被冲刷——即使某条整数指令已经乱序执行完，
  其结果也只存在于窗口标签里，随冲刷一并回收（RAT 还原），PC 改向目标。
* **精确异常**：非法 LOAD/STORE 在执行时识别（不触碰内存），延迟到
  退休才报告 :class:`Fault.ADDR`；当周期机器停止，年轻窗口条目全部丢弃，
  语义与顺序参考及旧流水线一致。

每周期相位顺序（确定性）：
  1. 退休（队首已完成则提交 1 条；故障在此报告）；
  2. 取指/重命名（把窗口填满，沿不命中路径，遇到 HALT 停止取指）；
  3. 整数发射（1 条最老就绪的算术指令，允许越过未解析分支推测执行）；
  4. 分支解析（至多 1 条最老的就绪 BEQ；命中则冲刷+重定向）；
  5. 顺序访存（队首 LOAD/STORE 就绪则执行 1 条，STORE 仍只退休才写）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .isa import (
    MASK32,
    MAX_CYCLES,
    NUM_REGS,
    CycleLimitError,
    Fault,
    Instruction,
    Op,
    validate_program,
)
from .pipeline import RetireRecord

# ---------------------------------------------------------------------------
# 对外常量与数据结构
# ---------------------------------------------------------------------------

REORDER_WINDOW = 4
"""默认重排序窗口大小（同时是飞行中指令数与物理标签数上限）。"""

MIN_WINDOW = 1
MAX_WINDOW = 16

_ALU_OPS = frozenset({Op.ADD, Op.SUB, Op.ADDI})


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    """一条指令"乱序完成"（执行结束，尚未退休）的记录。

    完成顺序就是整数口/访存口/分支口的实际服务顺序，可能与程序序不同；
    被分支冲刷的推测完成也留一条记录（``squashed=True``），用以展示
    "年轻结果先产生、后回收"。

    kind: "alu" / "load" / "store" / "branch"。
    """

    cycle: int
    pc: int
    op: Op
    kind: str
    value: Optional[int] = None
    addr: Optional[int] = None
    bad: bool = False
    taken: Optional[bool] = None
    squashed: bool = False


@dataclass(frozen=True, slots=True)
class WindowSlot:
    """周期末重排序窗口中的一个条目。

    state: "busy"（已重命名、未完成）或 "done"（已完成、等待退休）。
    """

    tag: int
    pc: int
    op: Op
    state: str


@dataclass(frozen=True, slots=True)
class WindowCycle:
    """一个周期的窗口快照与控制事件。

    fetch_pc: 本周期取指相位结束后下一条待取指令下标。
    squashed:  本周期因分支命中被冲刷的年轻指令下标（动态序，可能重复）。
    redirect:  分支命中后的取指目标下标；未重定向为 None。
    """

    cycle: int
    fetch_pc: int
    slots: tuple[WindowSlot, ...]
    squashed: tuple[int, ...]
    redirect: Optional[int] = None


@dataclass(frozen=True, slots=True)
class OoORunResult:
    """一次乱序核心运行的完整结果。"""

    regs: tuple[int, ...]
    memory: bytes
    trace: tuple[RetireRecord, ...]
    completions: tuple[CompletionRecord, ...]
    window: tuple[WindowCycle, ...]
    halted: bool
    fault: Optional[Fault]
    fault_pc: Optional[int]
    fault_cycle: Optional[int]

    @property
    def cycles_used(self) -> int:
        return len(self.window)


# ---------------------------------------------------------------------------
# 内部窗口条目
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Entry:
    tag: int
    pc: int
    insn: Instruction
    # 两个源操作数：tag 非 None 表示等待窗口中更老条目的结果，
    # tag 为 None 时使用译码时从架构寄存器堆抓取的快照值。
    s1_tag: Optional[int]
    s1_val: int
    s2_tag: Optional[int]
    s2_val: int
    old_map: Optional[int]          # 目的寄存器重命名前的旧标签（冲刷还原用）
    done: bool = False
    result: int = 0                 # ALU 结果 / LOAD 装入值
    store_data: int = 0             # STORE 待存值（执行期解析，退休期写入）
    addr: int = 0
    bad_addr: bool = False
    taken: Optional[bool] = None    # 仅 BEQ：解析结果


# ---------------------------------------------------------------------------
# 周期内核
# ---------------------------------------------------------------------------


def run_out_of_order(
    instructions: list[Instruction] | tuple[Instruction, ...],
    memory: bytes | bytearray,
    initial_regs: Optional[list[int]] = None,
    *,
    window: int = REORDER_WINDOW,
) -> OoORunResult:
    """在 4 项重排序窗口 + 寄存器重命名的乱序核心上执行程序。

    静态契约（程序/内存/初值）与 :func:`tinycpu.pipeline.run` 完全相同，
    非法输入在任何周期之前抛出，工作内存使用校验副本，无半更新状态。

    Args:
        window: 重排序窗口项数，1..16 的整数（``bool`` 显式拒绝）。
            默认 :data:`REORDER_WINDOW` = 4；取 1 时退化为按序执行。

    Raises:
        TypeError/ValueError/ProgramError: 输入不合法（含 window 越界）。
        CycleLimitError: 跑满 MAX_CYCLES 仍未结束。
    """
    prog, mem, regs = validate_program(instructions, memory, initial_regs)
    window = _check_window(window)

    # RAT：架构寄存器 -> 产生其最新值的存活窗口条目标签；None 表示该寄存器
    # 的最新值已在架构寄存器堆中（译码时直接快照 regs）。
    rat: list[Optional[int]] = [None] * NUM_REGS
    rob: list[_Entry] = []
    by_tag: dict[int, _Entry] = {}

    fetch_pc = 0
    fetch_halted = False          # HALT 已读入窗口，不再取指
    next_tag = 0

    trace: list[RetireRecord] = []
    completions: list[CompletionRecord] = []
    window_log: list[WindowCycle] = []

    halted = False
    fault: Optional[Fault] = None
    fault_pc: Optional[int] = None
    fault_cycle: Optional[int] = None

    for cycle in range(1, MAX_CYCLES + 1):
        squashed: list[int] = []
        redirect: Optional[int] = None

        # --------------------------------------------------------------
        # 相位 1：按序退休（队首完成则提交 1 条）。
        # --------------------------------------------------------------
        if rob:
            h = rob[0]
            if h.done:
                op = h.insn.op
                if op in (Op.LOAD, Op.STORE) and h.bad_addr:
                    # 非法访存延迟到退休才报告：本指令不提交，窗口中更年轻
                    # 的指令全部丢弃（即使某条算术已乱序完成，其结果只在标签
                    # 中，架构寄存器从未写入）；年轻 STORE 受顺序访存约束从未
                    # 执行，不可能写过内存。
                    fault = Fault.ADDR
                    fault_pc = h.pc
                    fault_cycle = cycle
                    trace.append(
                        RetireRecord(cycle=cycle, pc=h.pc, op=op, fault=Fault.ADDR)
                    )
                    by_tag.pop(h.tag, None)
                    del rob[0]
                    window_log.append(
                        WindowCycle(cycle, fetch_pc, (), (), None)
                    )
                    break

                kw: dict = {"cycle": cycle, "pc": h.pc, "op": op}
                if op in (Op.ADD, Op.SUB, Op.ADDI):
                    if h.insn.rd != 0:
                        regs[h.insn.rd] = h.result
                        if rat[h.insn.rd] == h.tag:
                            rat[h.insn.rd] = None
                        kw.update(rd=h.insn.rd, value=h.result)
                elif op == Op.LOAD:
                    if h.insn.rd != 0:
                        regs[h.insn.rd] = h.result
                        if rat[h.insn.rd] == h.tag:
                            rat[h.insn.rd] = None
                        kw.update(rd=h.insn.rd, value=h.result)
                elif op == Op.STORE:
                    mem[h.addr : h.addr + 4] = h.store_data.to_bytes(4, "little")
                    kw.update(addr=h.addr, mem_value=h.store_data)
                elif op == Op.HALT:
                    halted = True
                trace.append(RetireRecord(**kw))

                # 弹出队首前，把更年轻条目中对本标签的等待固化为具体值，
                # 之后标签才可释放。
                for y in rob[1:]:
                    if y.s1_tag == h.tag:
                        y.s1_tag, y.s1_val = None, h.result
                    if y.s2_tag == h.tag:
                        y.s2_tag, y.s2_val = None, h.result
                by_tag.pop(h.tag, None)
                del rob[0]

                if halted:
                    window_log.append(
                        WindowCycle(cycle, fetch_pc, (), (), None)
                    )
                    break

        # --------------------------------------------------------------
        # 相位 2：取指 + 重命名，沿"不命中"路径把窗口填满。
        # --------------------------------------------------------------
        while len(rob) < window and not fetch_halted:
            pc = fetch_pc
            insn = prog[pc]
            tag = next_tag
            next_tag += 1

            s1_tag, s1_val = _rename_src(insn.rs1, rat, regs)
            s2_tag, s2_val = _rename_src(insn.rs2, rat, regs)

            old_map: Optional[int] = None
            if insn.writes_rd and insn.rd != 0:
                old_map = rat[insn.rd]
                rat[insn.rd] = tag

            entry = _Entry(
                tag=tag, pc=pc, insn=insn,
                s1_tag=s1_tag, s1_val=s1_val,
                s2_tag=s2_tag, s2_val=s2_val,
                old_map=old_map,
            )
            # HALT 无执行动作：读入窗口即"完成"，排到队首即可退休。
            if insn.op == Op.HALT:
                entry.done = True
            rob.append(entry)
            by_tag[tag] = entry

            fetch_pc += 1
            if insn.op == Op.HALT:
                fetch_halted = True
                break
            # 静态校验保证程序以 HALT 结尾、BEQ 目标在程序内，顺序掉出末尾
            # 在此子集里不可能发生。

        # --------------------------------------------------------------
        # 相位 3：整数发射——1 条最老的就绪算术指令（可越过未解析分支）。
        # --------------------------------------------------------------
        for e in rob:
            if e.done or e.insn.op not in _ALU_OPS:
                continue
            if _ready(e, by_tag):
                v1 = _operand(e, True, by_tag)
                v2 = _operand(e, False, by_tag)
                op = e.insn.op
                if op == Op.ADD:
                    e.result = (v1 + v2) & MASK32
                elif op == Op.SUB:
                    e.result = (v1 - v2) & MASK32
                else:  # ADDI
                    e.result = (v1 + e.insn.imm) & MASK32
                e.done = True
                completions.append(
                    CompletionRecord(cycle=cycle, pc=e.pc, op=op, kind="alu",
                                     value=e.result)
                )
                break

        # --------------------------------------------------------------
        # 相位 4：分支解析——至多 1 条；更老的未解析 BEQ 优先（按序解析）。
        # --------------------------------------------------------------
        for e in rob:
            if e.insn.op != Op.BEQ or e.done:
                continue
            # 这是最老的未解析分支；它没就绪就不看更年轻的分支。
            if not _ready(e, by_tag):
                break
            v1 = _operand(e, True, by_tag)
            v2 = _operand(e, False, by_tag)
            taken = v1 == v2
            e.taken = taken
            e.done = True
            completions.append(
                CompletionRecord(cycle=cycle, pc=e.pc, op=Op.BEQ,
                                 kind="branch", taken=taken)
            )
            if taken:
                target = e.pc + e.insn.imm  # 已静态校验落在 [0, n)
                redirect = target
                # 冲刷所有更年轻条目（必为窗口后缀）：最年轻 -> 最老还原 RAT；
                # 已经乱序完成的算术结果随标签一并回收。
                idx = rob.index(e)
                for y in reversed(rob[idx + 1 :]):
                    squashed.append(y.pc)
                    completions.append(
                        CompletionRecord(
                            cycle=cycle, pc=y.pc, op=y.insn.op,
                            kind=_kind_of(y.insn.op),
                            value=y.result if y.done else None,
                            addr=y.addr if y.done else None,
                            bad=y.bad_addr if y.done else False,
                            squashed=True,
                        )
                    )
                    if (
                        y.insn.writes_rd
                        and y.insn.rd != 0
                        and rat[y.insn.rd] == y.tag
                    ):
                        rat[y.insn.rd] = y.old_map
                    by_tag.pop(y.tag, None)
                del rob[idx + 1 :]
                fetch_pc = target
                fetch_halted = False
            break

        # --------------------------------------------------------------
        # 相位 5：顺序访存——仅队首 LOAD/STORE 且操作数就绪才执行。
        # --------------------------------------------------------------
        if rob:
            h = rob[0]
            if (
                not h.done
                and h.insn.op in (Op.LOAD, Op.STORE)
                and _ready(h, by_tag)
            ):
                base = _operand(h, True, by_tag)
                h.addr = (base + h.insn.imm) & MASK32
                h.bad_addr = not _addr_ok(h.addr, len(mem))
                if h.insn.op == Op.LOAD:
                    if not h.bad_addr:
                        h.result = int.from_bytes(
                            mem[h.addr : h.addr + 4], "little"
                        )
                    h.done = True
                    completions.append(
                        CompletionRecord(
                            cycle=cycle, pc=h.pc, op=Op.LOAD, kind="load",
                            value=h.result if not h.bad_addr else None,
                            addr=h.addr, bad=h.bad_addr,
                        )
                    )
                else:  # STORE：解析地址与数据，但不写内存（退休才写）
                    if not h.bad_addr:
                        h.store_data = _operand(h, False, by_tag)
                    h.done = True
                    completions.append(
                        CompletionRecord(
                            cycle=cycle, pc=h.pc, op=Op.STORE, kind="store",
                            value=h.store_data if not h.bad_addr else None,
                            addr=h.addr, bad=h.bad_addr,
                        )
                    )

        window_log.append(
            WindowCycle(
                cycle,
                fetch_pc,
                tuple(
                    WindowSlot(x.tag, x.pc, x.insn.op, "done" if x.done else "busy")
                    for x in rob
                ),
                tuple(squashed),
                redirect,
            )
        )

    else:
        raise CycleLimitError(
            f"乱序核心在 {MAX_CYCLES} 个周期内未结束（可能存在 BEQ 死循环）"
        )

    return OoORunResult(
        regs=tuple(regs),
        memory=bytes(mem),
        trace=tuple(trace),
        completions=tuple(completions),
        window=tuple(window_log),
        halted=halted,
        fault=fault,
        fault_pc=fault_pc,
        fault_cycle=fault_cycle,
    )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _check_window(window: object) -> int:
    if isinstance(window, bool):
        raise TypeError("window 必须是 int，不能是 bool")
    if not isinstance(window, int):
        raise TypeError(f"window 必须是 int，收到 {type(window).__name__}")
    if not MIN_WINDOW <= window <= MAX_WINDOW:
        raise ValueError(
            f"window 必须在 [{MIN_WINDOW}, {MAX_WINDOW}]，收到 {window}"
        )
    return window


def _rename_src(
    reg: int,
    rat: list[Optional[int]],
    regs: list[int],
) -> tuple[Optional[int], int]:
    """译码期读源：返回 (等待标签或 None, 架构值快照)。"""
    if reg == 0:
        return None, 0
    tag = rat[reg]
    if tag is None:
        return None, regs[reg]
    # 标签必然指向窗口中更老的存活条目。
    return tag, 0


def _ready(e: _Entry, by_tag: dict[int, _Entry]) -> bool:
    if e.s1_tag is not None and not by_tag[e.s1_tag].done:
        return False
    if e.s2_tag is not None and not by_tag[e.s2_tag].done:
        return False
    return True


def _operand(e: _Entry, first: bool, by_tag: dict[int, _Entry]) -> int:
    tag = e.s1_tag if first else e.s2_tag
    if tag is None:
        return e.s1_val if first else e.s2_val
    return by_tag[tag].result


def _addr_ok(addr: int, mem_len: int) -> bool:
    return addr % 4 == 0 and 0 <= addr <= mem_len - 4


def _kind_of(op: Op) -> str:
    if op in _ALU_OPS:
        return "alu"
    if op == Op.LOAD:
        return "load"
    if op == Op.STORE:
        return "store"
    return "branch"
