"""两位饱和计数器分支预测器（直接索引）与预测统计。

预测器与流水线内核解耦：tinycpu.pipeline.run 接受可选的
predictor=BranchPredictor(...) 参数；不传时行为与基线完全一致
（等价于"永远预测不命中"）。

设计契约：
* 表项为 2 位饱和计数器（0..3），>=2 预测命中（taken），否则不命中；
* 直接索引：index = pc % entries，不同 PC 允许别名碰撞，不做标记匹配；
* 表容量 entries 与计数器初值 initial 由调用方给定；
* 训练只应由**真正提交**（WB 退休）的分支触发——update 由流水线在
  WB 级调用；被冲刷的分支不会到达 WB，因此不会污染计数器。
"""

from __future__ import annotations

from dataclasses import dataclass

from .isa import check_int

COUNTER_MIN = 0
COUNTER_MAX = 3
# 2 位饱和计数器的"命中"阈值：10/11 预测 taken。
_TAKEN_THRESHOLD = 2


class BranchPredictor:
    """直接索引的两位饱和计数器预测表。

    参数：
        entries: 表项数（>=1）。不同 PC 以 pc % entries 取模共用表项，
            允许别名碰撞，这正是"直接索引"的语义。
        initial: 每个计数器的初值（0..3），调用方给定。
    """

    __slots__ = ("_counters",)

    def __init__(self, entries: int, initial: int = 1) -> None:
        entries = check_int("entries", entries)
        if entries < 1:
            raise ValueError(f"entries 必须 >= 1，收到 {entries}")
        initial = check_int("initial", initial)
        if not COUNTER_MIN <= initial <= COUNTER_MAX:
            raise ValueError(
                f"initial 必须是 2 位计数器值 [{COUNTER_MIN}, {COUNTER_MAX}]，收到 {initial}"
            )
        self._counters: list[int] = [initial] * entries

    @property
    def entries(self) -> int:
        return len(self._counters)

    @property
    def counters(self) -> tuple[int, ...]:
        """当前全部计数器值的快照（只读副本）。"""
        return tuple(self._counters)

    def _index(self, pc: int) -> int:
        pc = check_int("pc", pc)
        if pc < 0:
            raise ValueError(f"pc 必须 >= 0，收到 {pc}")
        return pc % len(self._counters)

    def counter(self, pc: int) -> int:
        """某 PC 对应表项的当前计数器值。"""
        return self._counters[self._index(pc)]

    def predict(self, pc: int) -> bool:
        """取指阶段查询：True 表示预测分支命中（taken）。"""
        return self._counters[self._index(pc)] >= _TAKEN_THRESHOLD

    def update(self, pc: int, taken: bool) -> None:
        """分支提交时训练：按真实方向饱和加/减计数器。

        只应由流水线在 WB 退休时调用；错误路径上被冲刷的分支不得调用。
        """
        if not isinstance(taken, bool):
            raise TypeError(f"taken 必须是 bool，收到 {type(taken).__name__}")
        i = self._index(pc)
        c = self._counters[i]
        if taken:
            self._counters[i] = min(c + 1, COUNTER_MAX)
        else:
            self._counters[i] = max(c - 1, COUNTER_MIN)


@dataclass(frozen=True, slots=True)
class PredictionStats:
    """一次带预测运行的统计（未启用预测时 RunResult.prediction 为 None）。

    predictions:    取指阶段做出的方向预测次数（含错误路径上稍后被冲刷、
                    未来得及解析的取指——预测确实发生并影响了取指）。
    mispredictions: EX 解析时发现方向/目标错误的次数（每次都伴随冲刷）。
    flushed:        因错判而被冲刷的年轻指令条数（ID 与 IF 两级受害者合计）。
    """

    predictions: int
    mispredictions: int
    flushed: int

