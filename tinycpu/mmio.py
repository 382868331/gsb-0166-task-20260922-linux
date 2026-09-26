"""显式 MMIO 地址范围与本地小状态机设备模型。

本模块不连接任何真实外设：设备用纯内存状态机模拟，完全离线。

* **读寄存器**（``read_addr``）：每次读弹出内部队首元素；队列为空时
  抛 :class:`MMIOEmptyError`，由流水线/顺序机转成精确异常
  ``Fault.MMIO_EMPTY`` 在提交点报告。
* **写寄存器**（``write_addr``）：每次写把 32 位值追加到写记录。

两个地址构成显式 MMIO 范围（见 :attr:`MMIOBus.address_range`）。落在范围
内的 LOAD/STORE 是设备访问而不是内存访问；设备访问**只在提交点**（流水线
WB 级 / 顺序机该指令执行时）发生，因此被分支冲刷或异常丢弃的年轻指令
物理上不可能触碰设备，设备操作对每条退休指令恰好发生一次。
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

from .isa import MASK32, TinyCPUError, check_int

# RetireRecord.mmio / SeqTraceEntry.mmio 的取值。
MMIO_READ = "read"
MMIO_WRITE = "write"


class MMIOEmptyError(TinyCPUError):
    """设备读寄存器队列为空时由设备抛出（内核转为 Fault.MMIO_EMPTY）。"""


class MMIOBus:
    """一条显式地址范围的 MMIO 总线，挂一个读寄存器和一个写寄存器。

    参数:
        read_addr: 读寄存器字节地址（4 对齐）。LOAD 命中此地址时弹出队首。
        write_addr: 写寄存器字节地址（4 对齐，且不等于 read_addr）。
            STORE 命中此地址时追加写记录。
        read_values: 读寄存器初始队列（32 位无符号整数序列，先进先出）。

    方向不匹配（LOAD 写寄存器 / STORE 读寄存器）或地址未对齐的访问，
    由内核按 ``Fault.ADDR`` 处理；空队列读按 ``Fault.MMIO_EMPTY`` 处理。
    """

    __slots__ = ("read_addr", "write_addr", "_queue", "read_log", "write_log")

    def __init__(
        self,
        read_addr: int,
        write_addr: int,
        read_values: Sequence[int] = (),
    ) -> None:
        ra = check_int("read_addr", read_addr)
        wa = check_int("write_addr", write_addr)
        for name, a in (("read_addr", ra), ("write_addr", wa)):
            if a % 4 != 0 or not 0 <= a <= MASK32 - 3:
                raise ValueError(
                    f"{name} 必须是 4 对齐的 32 位字节地址，收到 {a}"
                )
        if ra == wa:
            raise ValueError("read_addr 与 write_addr 不能相同")
        if isinstance(read_values, (str, bytes)) or not isinstance(
            read_values, Sequence
        ):
            raise TypeError("read_values 必须是 32 位无符号整数序列")
        q: deque[int] = deque()
        for i, v in enumerate(read_values):
            iv = check_int(f"read_values[{i}]", v)
            if not 0 <= iv <= MASK32:
                raise ValueError(
                    f"read_values[{i}] 必须是 32 位无符号数，收到 {iv}"
                )
            q.append(iv)
        self.read_addr = ra
        self.write_addr = wa
        self._queue = q
        # 提交日志：设备实际被访问的记录（只增不减，供测试/演示核对）。
        self.read_log: list[int] = []
        self.write_log: list[int] = []

    # -- 地址译码 -----------------------------------------------------------

    @property
    def address_range(self) -> tuple[int, int]:
        """显式 MMIO 字节范围 [lo, hi)（含两个寄存器之间的空洞）。"""
        lo = min(self.read_addr, self.write_addr)
        hi = max(self.read_addr, self.write_addr) + 4
        return (lo, hi)

    def contains(self, addr: int) -> bool:
        """addr 是否命中任一设备寄存器。"""
        return addr == self.read_addr or addr == self.write_addr

    def is_read_reg(self, addr: int) -> bool:
        return addr == self.read_addr

    def is_write_reg(self, addr: int) -> bool:
        return addr == self.write_addr

    # -- 设备状态机（只能由内核在提交点调用） -------------------------------

    def read(self) -> int:
        """弹出读队列队首；空队列抛 :class:`MMIOEmptyError`。"""
        if not self._queue:
            raise MMIOEmptyError(
                f"MMIO 读寄存器 0x{self.read_addr:08X} 队列为空"
            )
        v = self._queue.popleft()
        self.read_log.append(v)
        return v

    def write(self, value: int) -> None:
        """把一次 32 位写追加到写记录。"""
        self.write_log.append(value & MASK32)

    # -- 观测 ---------------------------------------------------------------

    def snapshot(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """(剩余读队列, 读日志, 写记录)，用于流水线/顺序机对照。"""
        return (tuple(self._queue), tuple(self.read_log), tuple(self.write_log))
