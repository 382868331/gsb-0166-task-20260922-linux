"""MMIO 设备寄存器演示（确定性，无网络/无 sleep/无预录结果）。

运行：python demo_iteration.py

内容：
1. 正常程序：设备读弹队列 -> 值写回并参与运算 -> 设备写追加记录，
   含同地址连续操作与 HALT 前最后一次写；打印提交日志中的实际访问点；
2. 分支冲刷：错误路径上的设备读/写被冲刷，设备状态零变化；
3. 设备空读错误：Fault.DEV 在 WB 报告，年轻设备写不发生；
每个场景都与独立顺序参考解释器逐条对照设备队列、写记录与寄存器。
"""

from __future__ import annotations

from tinycpu import (
    Fault,
    Instruction,
    MMIO_READ_ADDR,
    MMIO_WRITE_ADDR,
    MMIODevices,
    memory_from_words,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

WIDTH = 78
I = Instruction


def _bar(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("-" * WIDTH)


def _print_program(prog) -> None:
    for i, ins in enumerate(prog):
        print(f"  pc{i:<2} {ins}")


def _print_commit_log(result) -> None:
    print("提交日志（WB 退休顺序，含设备实际访问点）：")
    for t in result.trace:
        if t.fault:
            what = f"故障 {t.fault.value}（不提交）"
        elif t.mmio == "read":
            what = f"MMIO读  dev[0x{t.addr:06X}] -> 弹出 {t.value}，写回 r{t.rd}"
        elif t.mmio == "write":
            what = f"MMIO写  dev[0x{t.addr:06X}] <- {t.mem_value}（追加写记录）"
        elif t.rd is not None:
            what = f"r{t.rd} <- {t.value}"
        elif t.addr is not None:
            what = f"mem[{t.addr}] <- {t.mem_value}"
        elif t.op.name == "HALT":
            what = "停机"
        else:
            what = "（无架构写）"
        print(f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {what}")


def _check_match(pipe, seq, dp, ds) -> None:
    assert pipe.regs == seq.regs, "寄存器不一致"
    assert pipe.memory == seq.memory, "内存不一致"
    assert dp.pending_reads == ds.pending_reads, "设备读队列不一致"
    assert dp.write_log == ds.write_log, "设备写记录不一致"
    assert [ (t.pc, t.mmio, t.value, t.mem_value, t.fault) for t in pipe.trace ] == \
           [ (t.pc, t.mmio, t.value, t.mem_value, t.fault) for t in seq.trace ], "退休轨迹不一致"
    print("与独立顺序参考解释器核对：寄存器 / 内存 / 设备队列 / 写记录 / 退休轨迹一致 ✔")


def demo_normal() -> None:
    _bar("演示 1（正常）：设备读->运算->设备写，同地址连续操作，HALT 前最后一写")
    prog = [
        I.addi(1, 0, MMIO_READ_ADDR),    # pc0  r1 = 设备读寄存器地址
        I.addi(2, 0, MMIO_WRITE_ADDR),   # pc1  r2 = 设备写寄存器地址
        I.load(3, 1, 0),                 # pc2  r3 = 弹出队首 41
        I.load(4, 1, 0),                 # pc3  同地址连续读：r4 = 弹出 7
        I.addi(5, 3, 1),                 # pc4  r5 = r3 + 1（load-use 停顿）
        I.add(6, 3, 4),                  # pc5  r6 = 41 + 7 = 48（前递）
        I.store(5, 2, 0),                # pc6  设备写 42
        I.store(6, 2, 0),                # pc7  同地址连续写：设备写 48
        I.store(6, 0, 0),                # pc8  普通内存写 mem[0] = 48（顺序对比）
        I.store(4, 2, 0),                # pc9  HALT 前最后一次设备写：7
        I.halt(),                        # pc10
    ]
    _print_program(prog)
    mem = memory_from_words([0] * 4)
    dp = MMIODevices([41, 7])
    ds = MMIODevices([41, 7])
    pipe = run(prog, bytearray(mem), devices=dp)
    seq = run_sequential(prog, bytearray(mem), devices=ds)

    _print_commit_log(pipe)
    print(f"\n最终寄存器：r3={pipe.regs[3]} r4={pipe.regs[4]} "
          f"r5={pipe.regs[5]} r6={pipe.regs[6]}（期望 41/7/42/48）")
    print(f"设备读队列剩余：{dp.pending_reads}（两个值都被恰好弹一次）")
    print(f"设备写记录：{dp.write_log}（期望 (42, 48, 7)，与程序顺序一致）")
    print(f"普通内存 mem[0..3] = {int.from_bytes(pipe.memory[0:4], 'little')}（期望 48）")
    _check_match(pipe, seq, dp, ds)


def demo_flush() -> None:
    _bar("演示 2（分支冲刷）：错误路径上的设备读/写不得发生")
    prog = [
        I.addi(1, 0, MMIO_READ_ADDR),    # pc0
        I.addi(2, 0, MMIO_WRITE_ADDR),   # pc1
        I.addi(3, 0, 5),                 # pc2  r3 = 5
        I.beq(3, 3, 3),                  # pc3  恒真 -> pc6，冲刷 pc4/pc5
        I.load(4, 1, 0),                 # pc4  被冲刷：不得弹队列
        I.store(3, 2, 0),                # pc5  被冲刷：不得追加写记录
        I.addi(5, 0, 9),                 # pc6  分支目标
        I.halt(),                        # pc7
    ]
    _print_program(prog)
    mem = memory_from_words([0] * 4)
    dp = MMIODevices([123])
    ds = MMIODevices([123])
    pipe = run(prog, bytearray(mem), devices=dp)
    seq = run_sequential(prog, bytearray(mem), devices=ds)

    _print_commit_log(pipe)
    print(f"\n退休指令下标：{[t.pc for t in pipe.trace]}（pc4/pc5 被冲刷，不出现）")
    print(f"设备读队列剩余：{dp.pending_reads}（123 仍在队列中，未被提前弹出）")
    print(f"设备写记录：{dp.write_log}（空，错误路径的写未发生）")
    _check_match(pipe, seq, dp, ds)


def demo_empty_read() -> None:
    _bar("演示 3（设备空读错误）：Fault.DEV 在 WB 报告，年轻设备写不发生")
    prog = [
        I.addi(1, 0, MMIO_READ_ADDR),    # pc0
        I.addi(2, 0, MMIO_WRITE_ADDR),   # pc1
        I.load(3, 1, 0),                 # pc2  队列已空 -> Fault.DEV
        I.store(3, 2, 0),                # pc3  年轻设备写，禁止发生
        I.halt(),                        # pc4
    ]
    _print_program(prog)
    mem = memory_from_words([0] * 4)
    dp = MMIODevices([])   # 空队列
    ds = MMIODevices([])
    pipe = run(prog, bytearray(mem), devices=dp)
    seq = run_sequential(prog, bytearray(mem), devices=ds)

    _print_commit_log(pipe)
    print(f"\n故障：{pipe.fault.value}，故障指令 pc{pipe.fault_pc}，"
          f"报告于周期 {pipe.fault_cycle}（WB 提交点）")
    print(f"r3 = {pipe.regs[3]}（故障读不写回，保持 0）")
    print(f"设备写记录：{dp.write_log}（pc3 的年轻写未发生）")
    assert pipe.fault is Fault.DEV
    _check_match(pipe, seq, dp, ds)


def main() -> None:
    demo_normal()
    print()
    demo_flush()
    print()
    demo_empty_read()
    print("=" * WIDTH)
    print("演示结束：设备为进程内小状态机，全部结果由 tinycpu 当场真实计算。")


if __name__ == "__main__":
    main()
