"""显式 MMIO 地址范围与本地设备小状态机。

地址映射（与数据内存严格分离，内存上限 1 MiB，MMIO 基址取 2 MiB）：

* MMIO_READ_ADDR  —— 设备读寄存器：LOAD 命中时**弹出**读队列队首；
  队列为空时产生 DeviceEmptyError，在 WB 以 Fault.DEV 报告。
* MMIO_WRITE_ADDR —— 设备写寄存器：STORE 命中时把 32 位值**追加**到写记录。

设备操作只在提交点（流水线 WB / 顺序机该指令退休时）执行且恰好一次：
被分支冲刷或异常丢弃的年轻指令不会触碰设备。设备读到的值与内存 LOAD
一样正常写回目标寄存器并参与前递，不存在"先预测后校正"。
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from .isa import MASK32, TinyCPUError, check_int

# 显式 MMIO 地址范围：两个 4 字节寄存器，落在数据内存上限（1 MiB）之外。
MMIO_BASE = 0x200000
MMIO_READ_ADDR = MMIO_BASE
MMIO_WRITE_ADDR = MMIO_BASE + 4
MMIO_END = MMIO_BASE + 8  # 半开区间 [MMIO_BASE, MMIO_END)


class DeviceEmptyError(TinyCPUError):
    """设备读队列为空时仍被读取（在提交点转化为 Fault.DEV）。"""


def is_mmio(addr: int) -> bool:
    """地址是否落在 MMIO 范围内。"""
    return MMIO_BASE <= addr < MMIO_END


def mmio_kind(addr: int) -> Optional[str]:
    """返回 "read" / "write" / None（非设备寄存器地址）。"""
    if addr == MMIO_READ_ADDR:
        return "read"
    if addr == MMIO_WRITE_ADDR:
        return "write"
    return None


class MMIODevices:
    """本地小状态机模拟的两个设备寄存器。

    * 读寄存器：内部一个 FIFO 队列，每次**已提交**的读弹出队首；
    * 写寄存器：每次**已提交**的写把值追加到写记录日志。

    不连接任何真实外设，全部状态在进程内。
    """

    __slots__ = ("_read_queue", "_write_log")

    def __init__(self, read_values: Iterable[int] = ()) -> None:
        if isinstance(read_values, (str, bytes, bytearray)) or not isinstance(
            read_values, Iterable
        ):
            raise TypeError("read_values 必须是 32 位无符号整数可迭代序列")
        self._read_queue: deque[int] = deque()
        for i, v in enumerate(read_values):
            iv = check_int(f"read_values[{i}]", v)
            if not 0 <= iv <= MASK32:
                raise ValueError(
                    f"read_values[{i}] 必须是 32 位无符号数 [0, 2^32-1]，收到 {iv}"
                )
            self._read_queue.append(iv)
        self._write_log: list[int] = []

    # -- 设备侧操作（只能由提交点调用） ------------------------------------

    def read(self) -> int:
        """弹出读队列队首；空队列抛 DeviceEmptyError。"""
        if not self._read_queue:
            raise DeviceEmptyError("设备读队列为空，无法弹出读寄存器值")
        return self._read_queue.popleft()

    def write(self, value: int) -> None:
        """把一次已提交的 32 位写追加到写记录。"""
        self._write_log.append(value & MASK32)

    # -- 观察状态（用于对照与测试） ----------------------------------------

    @property
    def pending_reads(self) -> tuple[int, ...]:
        """读队列中尚未弹出的值（队首在前）。"""
        return tuple(self._read_queue)

    @property
    def write_log(self) -> tuple[int, ...]:
        """已提交的设备写记录（按提交顺序）。"""
        return tuple(self._write_log)
