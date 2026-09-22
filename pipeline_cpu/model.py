"""公开数据结构、接口契约与错误语义。

机器模型
--------
* 32 个 32 位通用寄存器 r0..r31；r0 硬连线为 0，对它的写回被丢弃。
* 字可寻址内存，地址均为字节地址但必须 4 字节对齐，32 位字以小端序存放。
* 指令字（程序）与数据共用同一地址空间，从地址 0 开始连续存放。
* 算术按 32 位无符号回绕（two's complement 环绕），不检测溢出。
* BEQ 的偏移为有符号 21 位（字节偏移，会做 4 字节对齐校验）。

退休(retire)指指令到达 WB 阶段并提交其结构效果(写寄存器/提交异常)。
STORE 在 MEM 阶段写内存（在同周期 WB 之前完成）；非法 STORE 在 MEM
发现、到 WB 才作为故障退休（此时它尚未写内存，保证精确异常）。

输入契约(适用于本模块全部公开函数)
----------------------------------
* 所有数值参数必须是 ``int``；``bool`` 不是合法整数，拒绝。
* 浮点 ``NaN`` / ``Infinity`` 一律拒绝（本模型不存在浮点入参，
  传入 float 本身即为类型错误）。
* 非法输入抛 :class:`InvalidProgramError`，消息中携带可定位信息
  （指令下标、寄存器号、地址等），绝不静默纠正或截断。
* 模拟是纯函数：失败（输入错误或预算耗尽）时不产生任何部分结果对象，
  也不修改调用方传入的内存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

# ---- 接口边界（题面上限） ------------------------------------------------

MAX_PROGRAM_INSTRUCTIONS: Final[int] = 200
"""单次模拟最多 200 条指令。"""

MAX_CYCLES: Final[int] = 2000
"""流水线最多推进 2000 个周期；仍未停机则预算耗尽。"""

MAX_MEMORY_WORDS: Final[int] = 1 << 28
"""内存上限 = 2^30 字节（字地址空间的四分之一），防止误传天文数字。"""

REGISTER_COUNT: Final[int] = 32
WORD_BITS: Final[int] = 32
WORD_MASK: Final[int] = (1 << WORD_BITS) - 1
ADDR_MASK: Final[int] = WORD_MASK  # 字节地址同样是 32 位


class Opcode(str, Enum):
    """支持的 7 条指令。"""

    ADD = "ADD"      # rd = rs + rt
    SUB = "SUB"      # rd = rs - rt
    ADDI = "ADDI"    # rt = rs + imm(I 型语义: rt 目标, rs 源)
    LOAD = "LOAD"    # rt = MEM[rs + imm]
    STORE = "STORE"  # MEM[rs + imm] = rt
    BEQ = "BEQ"      # if rs == rt: pc += imm
    HALT = "HALT"    # 停机


@dataclass(frozen=True)
class Instruction:
    """一条指令。

    字段含义随操作数类型变化：

    ======= =================== =================== =================
    opcode  a                   b                   imm
    ======= =================== =================== =================
    ADD     rd(目标)            rs, rt 见 c/s       不用
    SUB     rd                  rs, rt 见 c/s       不用
    ADDI    rt(目标)            rs                  有符号立即数
    LOAD    rt(目标)            rs                  有符号偏移
    STORE   --                  rs(base), rt 见 c   有符号偏移
    BEQ     --                  rs, rt 见 c         有符号字节偏移
    HALT    --                  --                  --
    ======= =================== =================== =================

    为让数据结构单一且自描述，三寄存器型使用 ``a=rd, b=rs, c=rt``；
    立即数/访存型使用 ``a=rt, b=rs, imm=...``；STORE 待存寄存器用
    ``c=rt``；BEQ 两个比较寄存器为 ``b=rs, c=rt``。未用字段必须为 0。
    """

    opcode: Opcode
    a: int = 0
    b: int = 0
    c: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        # 校验集中在 ProgramSpec，这里只做轻量类型防呆。
        if not isinstance(self.opcode, Opcode):
            raise InvalidProgramError("Instruction.opcode 必须是 Opcode 枚举值")


@dataclass(frozen=True)
class Fault:
    """精确异常的退休记录。

    kind:
        ``"bad_address"``  -- LOAD/STORE 地址越界或未对齐
        ``"bad_fetch"``    -- 取指地址越界（程序顺序跑飞）
    """

    kind: str
    address: int


@dataclass(frozen=True)
class Retirement:
    """一条指令的退休轨迹项（按退休顺序，即 WB 先后）。"""

    seq: int | None
    """静态指令序号（在程序数组中的下标）；bad_fetch 故障无序号为 None。"""

    opcode: Opcode | None
    cycle: int
    """退休发生的周期号（从 1 开始）。"""

    rd: int | None
    """写回的目标寄存器编号；无写回或写 r0 仍给原编号，value 给 None。"""

    value: int | None
    """写回的 32 位值；无写回为 None。"""

    fault: Fault | None = None
    halted: bool = False


@dataclass(frozen=True)
class StageState:
    """某一周期某一流水槽的状态。"""

    seq: int | None
    """槽内指令的静态序号；气泡为 None。"""

    stalled: bool = False
    """该槽本周期是否被冻结（load-use 冒险时 ID 及其后方冻结）。"""

    flushed: bool = False
    """该槽指令是否为本周期被冲刷的年轻指令（分支决策冲刷）。"""


@dataclass(frozen=True)
class CycleSlots:
    """一个周期结束时五个流水槽的快照（IF/ID/EX/MEM/WB）。"""

    cycle: int
    if_stage: StageState
    id_stage: StageState
    ex_stage: StageState
    mem_stage: StageState
    wb_stage: StageState
    note: str = ""
    """本周期事件注记：``""`` / ``"stall"`` / ``"flush"`` / ``"stall+flush"``。"""


@dataclass(frozen=True)
class SimulationResult:
    """:func:`pipeline_cpu.core.run_pipeline` 的完整结果。

    ``retirements`` 包含正常退休指令与故障退休指令；故障项之后流水线中
    不再有更年轻指令退休（精确异常语义）。``halted`` 与 ``fault`` 互斥。
    """

    halted: bool
    fault: Fault | None
    fault_seq: int | None
    fault_cycle: int | None
    cycles: int
    retirements: tuple[Retirement, ...]
    trace: tuple[CycleSlots, ...]
    final_registers: tuple[int, ...]
    final_memory: bytes
    retired_count: int = field(default=0)


@dataclass(frozen=True)
class ReferenceResult:
    """:func:`pipeline_cpu.reference.run_sequential` 的结果。

    顺序模型在即将执行故障指令*之前*的寄存器/内存即“异常前”状态，
    与流水线在故障退休瞬间的结构状态逐值可比（流水线中更年轻指令
    尚未产生任何结构效果）。
    """

    halted: bool
    fault: Fault | None
    fault_seq: int | None
    executed: tuple[tuple[int, int | None, int | None, bool], ...]
    """(seq, rd, value, halted) 提交轨迹，无故障概念上的故障项。"""
    final_registers: tuple[int, ...]
    final_memory: bytes
    executed_count: int = field(default=0)


class InvalidProgramError(ValueError):
    """输入程序/参数非法。消息包含可定位信息，绝不静默纠正。"""


class CycleLimitExceeded(RuntimeError):
    """2000 周期预算耗尽（与“证明无解/停机”严格区分）。

    本 ISA 中 BEQ 偏移恒为静态常量、无动态写 PC，因此停机与否可静态
    判断；该异常只在模拟超过周期上限时抛出，表示预算不足而非程序死循环
    的“证明”。
    """


# ---- 输入校验 -------------------------------------------------------------


def _check_int(value: object, what: str) -> int:
    """整数参数校验：拒绝 bool，拒绝 float（含 NaN/Infinity），拒绝越界。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidProgramError(
            f"{what} 必须是 int，实际类型 {type(value).__name__}"
        )
    return value


def check_register(value: object, what: str) -> int:
    n = _check_int(value, what)
    if not 0 <= n < REGISTER_COUNT:
        raise InvalidProgramError(f"{what} 寄存器号越界: {n}（合法 0..31）")
    return n


def check_imm16(value: object, what: str = "imm") -> int:
    n = _check_int(value, what)
    if not -(1 << 15) <= n < (1 << 15):
        raise InvalidProgramError(f"{what} 超出有符号 16 位范围: {n}")
    return n


def check_branch_imm(value: object) -> int:
    n = _check_int(value, "BEQ 偏移")
    if not -(1 << 20) <= n < (1 << 20):
        raise InvalidProgramError(f"BEQ 偏移超出有符号 21 位范围: {n}")
    if n % 4 != 0:
        raise InvalidProgramError(f"BEQ 偏移必须 4 字节对齐: {n}")
    return n


def validate_instruction(inst: object, index: int) -> Instruction:
    """校验并规范化第 ``index`` 条指令，返回校验后的对象。"""
    if not isinstance(inst, Instruction):
        raise InvalidProgramError(
            f"程序[{index}] 必须是 Instruction，实际类型 {type(inst).__name__}"
        )
    op = inst.opcode
    where = f"程序[{index}] {op.value}"

    a = _check_int(inst.a, f"{where}.a")
    b = _check_int(inst.b, f"{where}.b")
    c = _check_int(inst.c, f"{where}.c")
    imm = _check_int(inst.imm, f"{where}.imm")

    if op in (Opcode.ADD, Opcode.SUB):
        check_register(a, f"{where} rd")
        check_register(b, f"{where} rs")
        check_register(c, f"{where} rt")
        if imm != 0:
            raise InvalidProgramError(f"{where} 不使用 imm，必须为 0，得到 {imm}")
    elif op == Opcode.ADDI:
        check_register(a, f"{where} rt")
        check_register(b, f"{where} rs")
        check_imm16(imm, f"{where} imm")
        if c != 0:
            raise InvalidProgramError(f"{where} 不使用 c，必须为 0，得到 {c}")
    elif op in (Opcode.LOAD,):
        check_register(a, f"{where} rt")
        check_register(b, f"{where} rs")
        check_imm16(imm, f"{where} imm")
        if c != 0:
            raise InvalidProgramError(f"{where} 不使用 c，必须为 0，得到 {c}")
    elif op == Opcode.STORE:
        check_register(b, f"{where} rs(base)")
        check_register(c, f"{where} rt(待存)")
        check_imm16(imm, f"{where} imm")
        if a != 0:
            raise InvalidProgramError(f"{where} 不使用 a，必须为 0，得到 {a}")
    elif op == Opcode.BEQ:
        check_register(b, f"{where} rs")
        check_register(c, f"{where} rt")
        check_branch_imm(imm)
        if a != 0:
            raise InvalidProgramError(f"{where} 不使用 a，必须为 0，得到 {a}")
    elif op == Opcode.HALT:
        if (a, b, c, imm) != (0, 0, 0, 0):
            raise InvalidProgramError(f"{where} 不使用操作数字段，必须全 0")
    else:  # pragma: no cover - 枚举已封闭
        raise InvalidProgramError(f"{where} 未知操作码 {op!r}")

    return Instruction(op, a, b, c, imm)


def validate_program(
    program: object, memory: object
) -> tuple[tuple[Instruction, ...], bytearray]:
    """校验程序数组与初始内存，返回(不可变程序副本, 内存可变副本)。

    程序长度 1..200；内存长度为 4 的倍数且不超过 2^30 字节；
    必须为可转字节的序列（bytes/bytearray）。调用方拿到的是副本，
    后续模拟不触碰调用方数据，校验失败时也无任何已分配状态外泄。
    """
    if not isinstance(program, (list, tuple)):
        raise InvalidProgramError(
            f"程序必须是 list/tuple[Instruction]，实际 {type(program).__name__}"
        )
    if len(program) == 0:
        raise InvalidProgramError("程序至少包含 1 条指令")
    if len(program) > MAX_PROGRAM_INSTRUCTIONS:
        raise InvalidProgramError(
            f"程序 {len(program)} 条超过上限 {MAX_PROGRAM_INSTRUCTIONS}"
        )
    prog = tuple(validate_instruction(x, i) for i, x in enumerate(program))

    if isinstance(memory, (bytes, bytearray)):
        mem = bytearray(memory)
    elif isinstance(memory, memoryview):
        if memory.format != "B":
            raise InvalidProgramError("memoryview 内存必须是字节格式(format='B')")
        mem = bytearray(memory)
    else:
        raise InvalidProgramError(
            f"内存必须是 bytes/bytearray，实际 {type(memory).__name__}"
        )
    if len(mem) % 4 != 0:
        raise InvalidProgramError(f"内存长度必须 4 字节对齐，得到 {len(mem)} 字节")
    if len(mem) > MAX_MEMORY_WORDS * 4:
        raise InvalidProgramError(
            f"内存 {len(mem)} 字节超过上限 {MAX_MEMORY_WORDS * 4}"
        )
    return prog, mem
