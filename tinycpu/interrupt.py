"""离线外部中断控制器（不连接真实 CPU，逐拍注入脉冲）。

本模块只有"事件源"语义，不含任何流水线知识：

* 中断向量是**固定**的 ``VECTOR_PC``（见 :mod:`tinycpu.isa`），不改动任何
  指令编码；``request`` 时显式给出的向量若不是该固定值，**先拒绝注入**
  （抛 ``ValueError``），不进入任何模拟周期；
* 每个脉冲只记"某拍有请求"。同一拍多个脉冲、屏蔽期间的多个脉冲，都
  **合并成一位 pending**，不计数；
* 脉冲按周期号登记，周期内核在该拍的退休边界采样（是否被接纳、与异常
  同拍谁优先等策略由执行器决定，控制器不关心）。

顺序参考解释器没有"周期"概念，它用 :class:`BoundaryInterrupts` 在相同的
*退休边界*上注入事件，两个执行器从而可以逐边界对齐比较。
"""

from __future__ import annotations

from dataclasses import dataclass

from .isa import MAX_CYCLES, VECTOR_PC, check_int


@dataclass(frozen=True, slots=True)
class InterruptAccept:
    """一次中断接纳的记录（退休轨迹之外的控制轨迹）。

    cycle: 接纳发生的周期（仅周期内核有；顺序参考为退休序号）。
    saved_pc: 被保存的"下一条尚未退休指令"下标，IRET 后从这里恢复。
    retired_pc: 接纳边界上正常退休的最老指令下标。
    """

    cycle: int
    saved_pc: int
    retired_pc: int


def _check_vector(vector: object) -> int:
    v = check_int("vector", vector)
    if v != VECTOR_PC:
        raise ValueError(
            f"非法中断向量 {v}：本实现固定向量 PC = {VECTOR_PC}，"
            f"不接受其它向量注入"
        )
    return v


class InterruptController:
    """周期级离线中断源：脉冲登记到具体周期，pending 合并为一位。"""

    def __init__(self, vector: int = VECTOR_PC) -> None:
        self._vector = _check_vector(vector)
        self._pulses: set[int] = set()

    @property
    def vector(self) -> int:
        return self._vector

    @property
    def pulses(self) -> frozenset[int]:
        """所有已登记脉冲周期（去重后；不暴露可写集合）。"""
        return frozenset(self._pulses)

    def request(self, cycle: int, vector: int = VECTOR_PC) -> "InterruptController":
        """登记一个在第 ``cycle`` 拍到达的中断脉冲。

        - ``cycle`` 必须是 [1, MAX_CYCLES] 的真整数（bool 拒绝）；
        - ``vector`` 必须等于固定 :data:`~tinycpu.isa.VECTOR_PC`，
          非法向量在注入时立即拒绝，不会登记任何脉冲；
        - 同拍重复请求合并为一位，不计数。
        """
        c = check_int("cycle", cycle)
        if not 1 <= c <= MAX_CYCLES:
            raise ValueError(f"cycle 必须在 [1, {MAX_CYCLES}]，收到 {c}")
        _check_vector(vector)
        self._pulses.add(c)
        return self

    # 语义别名：pulse 与 request 等价。
    pulse = request

    def has_pulse(self, cycle: int) -> bool:
        return cycle in self._pulses

    def __len__(self) -> int:
        return len(self._pulses)


class BoundaryInterrupts:
    """退休边界级离线中断源，供无流水顺序参考使用。

    ``request_after(k)`` 表示：第 k 条指令正常退休的那个边界上有脉冲
    （k 从 0 起：k=0 即第一条指令退休前/时的边界，等价于周期内核中
    第一条指令退休那一拍采样到的脉冲）。同一个 k 重复登记合并为一位。

    与周期内核的对齐由测试负责：小程序的退休周期可静态手算，
    脉冲拍号与退休序号一一对应（见 tests/test_interrupts.py）。
    """

    def __init__(self, vector: int = VECTOR_PC) -> None:
        self._vector = _check_vector(vector)
        self._pulses: set[int] = set()

    @property
    def vector(self) -> int:
        return self._vector

    @property
    def boundaries(self) -> frozenset[int]:
        return frozenset(self._pulses)

    def request_after(
        self, retired_count: int, vector: int = VECTOR_PC
    ) -> "BoundaryInterrupts":
        k = check_int("retired_count", retired_count)
        if k < 0:
            raise ValueError(f"retired_count 不能为负，收到 {k}")
        _check_vector(vector)
        self._pulses.add(k)
        return self

    def has_pulse(self, retired_count: int) -> bool:
        return retired_count in self._pulses

    def __len__(self) -> int:
        return len(self._pulses)
