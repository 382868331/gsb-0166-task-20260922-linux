"""显式中断请求与 IRET 的离线控制器模型、事件记录与共享校验。

本模块不含任何流水线时序。中断的接纳点、保存 PC、冲刷与解除屏蔽的
语义在 :mod:`tinycpu.pipeline`（五级流水）与 :mod:`tinycpu.sequential`
（顺序参考）中各自独立实现；本模块只提供两者共用的三件东西：

* :class:`InterruptController` —— 离线中断控制器模型。它只是一个
  纯软件状态机：1 位 pending 锁存，由执行器在每个**退休边界**步进，
  不连接真实 CPU、操作系统或任何外部硬件；
* :class:`IrqEvent` —— 一次中断接纳的记录（接纳边界、保存 PC、向量 PC）；
* :func:`check_interrupt_setup` —— 控制器/向量参数的静态校验，
  非法向量在任何周期执行之前拒绝注入。

语义契约（两个执行器完全一致）：

1. 中断只在**退休边界**评估：当前最老指令在 WB 正常退休之后。
2. 同一边界有同步访存异常时，异常优先，中断保持 pending。
3. 接纳时保存"下一条尚未退休指令的 PC"，冲刷全部更年轻指令，
   转向固定向量 PC，并屏蔽嵌套中断。
4. 屏蔽期间到达的请求仍锁存进 pending（最多 1 位，多脉冲合并，不计数）。
5. IRET 恢复保存的 PC；屏蔽在 IRET 退休之后才解除（IRET 自己的
   退休边界仍按"屏蔽中"评估，最早下一个边界才可再次接纳）。
6. HALT 退休后不再接纳中断；pending 保持，可在结果中观察到。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .isa import ProgramError, check_int


class InterruptController:
    """离线中断控制器：1 位 pending 锁存，逐退休边界由执行器步进。

    两种驱动方式（都在本进程内离线完成，不触碰真实 CPU）：

    * ``request()`` —— 外部脉冲，把 pending 置 1。运行前预置即可让
      第一个退休边界就观察到请求；多次调用合并为同一位，不计数。
    * ``pulse_boundaries`` —— 构造时给出的退休边界下标（0 起）集合；
      执行器在第 k 次退休的边界上自动到达一个脉冲。流水线与顺序参考
      用**同一个**边界编号注入，因此两者的注入点对齐、结果可逐位比较。

    属性 ``pending`` 可随时读取；请求在未被接纳前一直保持。
    """

    def __init__(self, pulse_boundaries: Iterable[int] = ()) -> None:
        if isinstance(pulse_boundaries, (str, bytes)) or not isinstance(
            pulse_boundaries, Iterable
        ):
            raise TypeError("pulse_boundaries 必须是非负整数可迭代对象")
        pulses: set[int] = set()
        for p in pulse_boundaries:
            v = check_int("pulse boundary", p)
            if v < 0:
                raise ValueError(f"退休边界下标必须非负，收到 {v}")
            pulses.add(v)
        self._pulses = frozenset(pulses)
        self.pending = False

    def request(self) -> None:
        """到达一个中断请求脉冲（多次脉冲合并为同一位）。"""
        self.pending = True

    # -- 执行器内部接口 -------------------------------------------------

    def _arrive(self, boundary: int) -> None:
        """第 boundary 次退休的边界：应用调度在该边界的脉冲。"""
        if boundary in self._pulses:
            self.pending = True

    def _take(self) -> None:
        """接纳中断：消费 pending 位。"""
        self.pending = False


@dataclass(frozen=True, slots=True)
class IrqEvent:
    """一次中断接纳的记录。

    boundary: 第几次退休（0 起）之后接纳，可直接索引退休轨迹；
    saved_pc: 保存的"下一条尚未退休指令"的 PC（IRET 将回到这里）；
    vector_pc: 转向的固定向量 PC；
    cycle: 流水线中的接纳周期；顺序参考没有周期概念，为 None。
    """

    boundary: int
    saved_pc: int
    vector_pc: int
    cycle: Optional[int] = None


def check_interrupt_setup(
    interrupts: object,
    interrupt_vector: object,
    prog_len: int,
) -> Optional[int]:
    """校验控制器/向量组合，返回向量 PC（未启用中断时返回 None）。

    非法向量（越界、非整数、bool 冒充）在任何周期执行之前拒绝注入；
    控制器与向量必须同时提供或同时省略。
    """
    if (interrupts is None) != (interrupt_vector is None):
        raise ProgramError(
            "interrupts 与 interrupt_vector 必须同时提供或同时省略"
        )
    if interrupts is None:
        return None
    if not isinstance(interrupts, InterruptController):
        raise TypeError(
            f"interrupts 必须是 InterruptController，收到 {type(interrupts).__name__}"
        )
    v = check_int("interrupt_vector", interrupt_vector)
    if not 0 <= v < prog_len:
        raise ProgramError(
            f"非法中断向量 {v}：必须落在程序下标 [0, {prog_len - 1}]，拒绝注入"
        )
    return v
