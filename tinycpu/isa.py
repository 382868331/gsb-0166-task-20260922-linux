"""ISA：指令格式、32 位语义常量、输入校验与（字节）内存辅助。

本模块只定义"静态"契约：指令如何构造、什么输入合法、内存如何布局。
流水线时序在 :mod:`tinycpu.pipeline`，顺序参考在 :mod:`tinycpu.sequential`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 体系结构常量
# ---------------------------------------------------------------------------

NUM_REGS = 32
MASK32 = 0xFFFFFFFF
MAX_INSTRUCTIONS = 200
MAX_CYCLES = 2000
# 数据内存的实现上限（题目未规定，给出明确边界而非放任分配）。
MAX_MEMORY_BYTES = 1 << 20
IMM_MIN = -(1 << 31)
IMM_MAX = (1 << 31) - 1
# 固定中断向量 PC：中断被接纳后下一条取指从该指令下标开始。
# 不引入任何指令编码/寻址改动；约定向量为 0，使用中断的程序把处理例程
# 放在 pc0 起，并用一条"冷启动跳过、进入处理例程时落下"的 BEQ 做引导
# （见 README 与测试）。
VECTOR_PC = 0


class Op(str, Enum):
    """八条指令。枚举值同时是对外可见的规范短名。"""

    ADD = "ADD"
    SUB = "SUB"
    ADDI = "ADDI"
    LOAD = "LOAD"
    STORE = "STORE"
    BEQ = "BEQ"
    HALT = "HALT"
    IRET = "IRET"


class Fault(str, Enum):
    """精确异常种类。

    ADDR: LOAD/STORE 有效地址未按 4 字节对齐或越出数据内存。
    异常在 WB 级报告（见 :mod:`tinycpu.pipeline`）。
    """

    ADDR = "ADDR"


# 短名常量，方便 ``from tinycpu import ADD``。
ADD = Op.ADD
SUB = Op.SUB
ADDI = Op.ADDI
LOAD = Op.LOAD
STORE = Op.STORE
BEQ = Op.BEQ
HALT = Op.HALT
IRET = Op.IRET


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class TinyCPUError(Exception):
    """本库所有自定义错误的基类。"""


class ProgramError(TinyCPUError, ValueError):
    """程序/内存/初值静态不合法（在任何模拟状态产生之前抛出）。"""


class InterruptStateError(TinyCPUError):
    """中断控制状态非法：如未处于中断处理时执行了 IRET。"""


class CycleLimitError(TinyCPUError):
    """运行超过周期预算（与"程序无解/异常"语义分离：这是预算耗尽）。"""


# ---------------------------------------------------------------------------
# 标量校验（拒绝 bool 冒充整数；拒绝 NaN/Infinity）
# ---------------------------------------------------------------------------


def check_int(name: str, value: object) -> int:
    """要求真正的整数。

    - ``bool`` 是 ``int`` 子类型，按题主要求显式拒绝；
    - ``float`` 中 NaN/+Inf/-Inf 报 :class:`ValueError` 语义错误，
      有限浮点仍不能当整数，报 :class:`TypeError`；
    - 其它非整数报 :class:`TypeError`。
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} 必须是 int，不能是 bool")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} 不能是 NaN 或 Infinity")
        raise TypeError(f"{name} 必须是 int，不能是 float")
    if isinstance(value, int):
        return value
    raise TypeError(f"{name} 必须是 int，收到 {type(value).__name__}")


def _check_reg(name: str, value: object) -> int:
    v = check_int(name, value)
    if not 0 <= v < NUM_REGS:
        raise ValueError(f"{name} 寄存器号必须在 [0, {NUM_REGS - 1}]，收到 {v}")
    return v


def _check_imm(name: str, value: object) -> int:
    v = check_int(name, value)
    if not IMM_MIN <= v <= IMM_MAX:
        raise ValueError(f"{name} 立即数必须是 32 位有符号数 [{IMM_MIN}, {IMM_MAX}]，收到 {v}")
    return v


# ---------------------------------------------------------------------------
# 指令
# ---------------------------------------------------------------------------

_WRITES_RD = frozenset({Op.ADD, Op.SUB, Op.ADDI, Op.LOAD})


@dataclass(frozen=True, slots=True)
class Instruction:
    """一条 32 位小 CPU 指令。

    字段语义按操作码解释（未使用字段保持默认 0）：

    ==== ======= ======= ======= =========================
    操作码 rd      rs1     rs2     imm
    ==== ======= ======= ======= =========================
    ADD  目标    源 1    源 2    -
    SUB  目标    源 1    源 2    -
    ADDI 目标    源 1    -       32 位有符号立即数
    LOAD 目标    基址    -       字节偏移（符号加）
    STORE -      基址    存数    字节偏移（符号加）
    BEQ  -       源 1    源 2    字偏移（PC 相对，符号）
    HALT -       -       -       -
    ==== ======= ======= ======= =========================

    BEQ 的 ``imm`` 以**字**为单位：目标指令下标 = 当前 pc + imm。
    所有算术按 32 位二进制定点回绕，寄存器中保存无符号位模式。
    """

    op: Op
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.op, Op):
            raise TypeError(
                f"op 必须是 tinycpu.Op 成员（如 tinycpu.ADD），收到 {self.op!r}"
            )
        op = self.op
        if op in (Op.ADD, Op.SUB):
            object.__setattr__(self, "rd", _check_reg("rd", self.rd))
            object.__setattr__(self, "rs1", _check_reg("rs1", self.rs1))
            object.__setattr__(self, "rs2", _check_reg("rs2", self.rs2))
        elif op == Op.ADDI:
            object.__setattr__(self, "rd", _check_reg("rd", self.rd))
            object.__setattr__(self, "rs1", _check_reg("rs1", self.rs1))
            object.__setattr__(self, "imm", _check_imm("imm", self.imm))
        elif op == Op.LOAD:
            object.__setattr__(self, "rd", _check_reg("rd", self.rd))
            object.__setattr__(self, "rs1", _check_reg("rs1", self.rs1))
            object.__setattr__(self, "imm", _check_imm("imm", self.imm))
        elif op == Op.STORE:
            object.__setattr__(self, "rs1", _check_reg("rs1", self.rs1))
            object.__setattr__(self, "rs2", _check_reg("rs2", self.rs2))
            object.__setattr__(self, "imm", _check_imm("imm", self.imm))
        elif op == Op.BEQ:
            object.__setattr__(self, "rs1", _check_reg("rs1", self.rs1))
            object.__setattr__(self, "rs2", _check_reg("rs2", self.rs2))
            # 程序最长 200 条，13 位以上的字偏移没有合法目标可能。
            object.__setattr__(self, "imm", _check_imm("imm", self.imm))
        # HALT/IRET：无字段需要校验。

    # -- 便捷构造器 ---------------------------------------------------------

    @classmethod
    def add(cls, rd: int, rs1: int, rs2: int) -> "Instruction":
        return cls(Op.ADD, rd=rd, rs1=rs1, rs2=rs2)

    @classmethod
    def sub(cls, rd: int, rs1: int, rs2: int) -> "Instruction":
        return cls(Op.SUB, rd=rd, rs1=rs1, rs2=rs2)

    @classmethod
    def addi(cls, rd: int, rs1: int, imm: int) -> "Instruction":
        return cls(Op.ADDI, rd=rd, rs1=rs1, imm=imm)

    @classmethod
    def load(cls, rd: int, base: int, offset: int) -> "Instruction":
        return cls(Op.LOAD, rd=rd, rs1=base, imm=offset)

    @classmethod
    def store(cls, src: int, base: int, offset: int) -> "Instruction":
        return cls(Op.STORE, rs1=base, rs2=src, imm=offset)

    @classmethod
    def beq(cls, rs1: int, rs2: int, offset_words: int) -> "Instruction":
        return cls(Op.BEQ, rs1=rs1, rs2=rs2, imm=offset_words)

    @classmethod
    def halt(cls) -> "Instruction":
        return cls(Op.HALT)

    @classmethod
    def iret(cls) -> "Instruction":
        return cls(Op.IRET)

    # -- 译码后属性 ---------------------------------------------------------

    @property
    def writes_rd(self) -> bool:
        return self.op in _WRITES_RD

    def __str__(self) -> str:
        """汇编风格文本，便于打印程序与轨迹。"""
        if self.op in (Op.ADD, Op.SUB):
            return f"{self.op.value} r{self.rd}, r{self.rs1}, r{self.rs2}"
        if self.op == Op.ADDI:
            return f"ADDI r{self.rd}, r{self.rs1}, {self.imm}"
        if self.op == Op.LOAD:
            return f"LOAD r{self.rd}, {self.imm}(r{self.rs1})"
        if self.op == Op.STORE:
            return f"STORE r{self.rs2}, {self.imm}(r{self.rs1})"
        if self.op == Op.BEQ:
            sign = "+" if self.imm >= 0 else ""
            return f"BEQ r{self.rs1}, r{self.rs2}, {sign}{self.imm}"
        if self.op == Op.IRET:
            return "IRET"
        return "HALT"


# ---------------------------------------------------------------------------
# 内存辅助：32 位字 <-> 小端字节
# ---------------------------------------------------------------------------


def memory_from_words(words: Sequence[int]) -> bytearray:
    """把 32 位无符号字序列编码成小端字节内存。"""
    if isinstance(words, (str, bytes, bytearray)) or not isinstance(
        words, Sequence
    ):
        raise TypeError("words 必须是整数序列（list/tuple）")
    mem = bytearray()
    for i, w in enumerate(words):
        v = check_int(f"words[{i}]", w)
        if not 0 <= v <= MASK32:
            raise ValueError(f"words[{i}] 必须是 32 位无符号数 [0, 2^32-1]，收到 {v}")
        mem += v.to_bytes(4, "little", signed=False)
    if len(mem) > MAX_MEMORY_BYTES:
        raise ProgramError(f"内存超过实现上限 {MAX_MEMORY_BYTES} 字节")
    return mem


def words_from_memory(memory: bytes | bytearray) -> list[int]:
    """把小端字节内存解码回 32 位无符号字列表（长度须为 4 的倍数）。"""
    if not isinstance(memory, (bytes, bytearray, memoryview)):
        raise TypeError("memory 必须是 bytes/bytearray")
    if len(memory) % 4 != 0:
        raise ValueError(f"内存长度必须是 4 的倍数，收到 {len(memory)}")
    return [
        int.from_bytes(memory[i : i + 4], "little", signed=False)
        for i in range(0, len(memory), 4)
    ]


# ---------------------------------------------------------------------------
# 程序整体静态校验（pipeline 与顺序参考共用，保证两者输入语义一致）
# ---------------------------------------------------------------------------


def validate_program(
    instructions: Sequence[Instruction],
    memory: bytes | bytearray,
    initial_regs: Optional[Sequence[int]] = None,
) -> tuple[tuple[Instruction, ...], bytearray, list[int]]:
    """校验程序、数据内存、寄存器初值，返回各自的可执行副本。

    任何一项不合法都抛 :class:`ProgramError`/``TypeError``/``ValueError``，
    且在第一条指令执行之前抛出——不存在"跑了一半再退回"的部分状态。
    """
    if isinstance(instructions, (str, bytes)) or not isinstance(
        instructions, Sequence
    ):
        raise TypeError("instructions 必须是 Instruction 序列（list/tuple）")
    n = len(instructions)
    if not 1 <= n <= MAX_INSTRUCTIONS:
        raise ProgramError(
            f"程序条数必须在 [1, {MAX_INSTRUCTIONS}]，收到 {n}"
        )
    prog: list[Instruction] = []
    for i, instr in enumerate(instructions):
        if not isinstance(instr, Instruction):
            raise TypeError(f"instructions[{i}] 必须是 Instruction，收到 {type(instr).__name__}")
        prog.append(instr)

    # 本 ISA 子集里程序必须以 HALT 结束（顺序执行到底没有别的定义）。
    if prog[-1].op != Op.HALT:
        raise ProgramError("程序最后一条指令必须是 HALT")

    # 分支目标静态可查（立即数固定），提前判非法，错误可定位到指令下标。
    for i, instr in enumerate(prog):
        if instr.op == Op.BEQ:
            target = i + instr.imm
            if not 0 <= target < n:
                raise ProgramError(
                    f"instructions[{i}] BEQ 目标越界：{i} + {instr.imm} = {target}，"
                    f"合法范围 [0, {n - 1}]"
                )

    if not isinstance(memory, (bytes, bytearray, memoryview)):
        raise TypeError("memory 必须是 bytes/bytearray")
    if len(memory) == 0 or len(memory) % 4 != 0:
        raise ProgramError(f"数据内存长度必须是正的 4 的倍数，收到 {len(memory)}")
    if len(memory) > MAX_MEMORY_BYTES:
        raise ProgramError(f"数据内存超过实现上限 {MAX_MEMORY_BYTES} 字节")
    mem = bytearray(memory)

    if initial_regs is None:
        regs = [0] * NUM_REGS
    else:
        if isinstance(initial_regs, (str, bytes)) or not isinstance(
            initial_regs, Sequence
        ):
            raise TypeError("initial_regs 必须是长度 32 的整数序列")
        if len(initial_regs) != NUM_REGS:
            raise ProgramError(f"initial_regs 长度必须为 {NUM_REGS}，收到 {len(initial_regs)}")
        regs = []
        for i, v in enumerate(initial_regs):
            iv = check_int(f"initial_regs[{i}]", v)
            if not 0 <= iv <= MASK32:
                raise ValueError(
                    f"initial_regs[{i}] 必须是 32 位无符号数 [0, 2^32-1]，收到 {iv}"
                )
            regs.append(iv)
        # r0 恒零：不静默写 0 覆盖，而是直接拒绝，错误可定位。
        if regs[0] != 0:
            raise ProgramError("initial_regs[0] 必须为 0（r0 恒零）")

    return tuple(prog), mem, regs
