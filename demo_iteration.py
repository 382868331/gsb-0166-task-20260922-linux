"""MMIO 增量演示：设备访问只在提交点发生恰好一次（真实当场计算）。

运行：python demo_iteration.py（无网络/无账号/无 sleep/无预录结果，
通常 0.1 秒内完成，远低于 90 秒限制）。

内容：
1. 正常程序：MMIO 读弹队首、写追加记录，打印提交日志中的实际访问点，
   并与独立顺序参考解释器逐条对照（寄存器/内存/设备队列/写记录）；
2. 分支冲刷：错误路径上的 MMIO 读/写被冲刷，设备状态零变化；
3. 设备空读：报 Fault.MMIO_EMPTY，老指令已提交、年轻 MMIO 写被丢弃。
"""

from __future__ import annotations

from tinycpu import Fault, Instruction, MMIOBus, memory_from_words
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

READ_ADDR = 0x00008000
WRITE_ADDR = 0x00008004
MEM = memory_from_words([0] * 8)
WIDTH = 78


def _bar(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("-" * WIDTH)


def _show_commit_log(result) -> None:
    print("提交日志（退休轨迹，cycle 即实际访问点）：")
    for r in result.trace:
        extra = ""
        if r.fault is not None:
            extra = f"  FAULT={r.fault.value}"
        elif r.mmio == "read":
            extra = f"  MMIO读  <- 0x{READ_ADDR:08X}  值={r.value}"
        elif r.mmio == "write":
            extra = f"  MMIO写  -> 0x{WRITE_ADDR:08X}  值={r.mem_value}"
        elif r.addr is not None:
            extra = f"  内存写  mem[0x{r.addr:04X}] = {r.mem_value}"
        elif r.rd is not None:
            extra = f"  r{r.rd} <- {r.value}"
        print(f"  cycle {r.cycle:>3}  pc={r.pc:<2} {r.op.value:<5}{extra}")


def _check_equiv(res, seq, bus_p, bus_s, label: str) -> None:
    ok = (
        res.regs == seq.regs
        and res.memory == seq.memory
        and res.fault == seq.fault
        and bus_p.snapshot() == bus_s.snapshot()
    )
    print(f"与顺序参考对照（寄存器/内存/故障/设备队列与写记录）：{'一致' if ok else '不一致！'}")
    assert ok, label


def main() -> None:
    _bar("1. 正常 MMIO 程序：读弹队首、写追加记录，恰好一次")
    prog = [
        Instruction.addi(1, 0, READ_ADDR),   # r1 = 读寄存器地址
        Instruction.load(2, 1, 0),           # r2 <- 弹出队首
        Instruction.load(3, 1, 0),           # r3 <- 再弹
        Instruction.store(3, 0, 0),          # 普通内存写 mem[0]
        Instruction.store(2, 1, 4),          # MMIO 写：把弹出的值写回设备
        Instruction.halt(),
    ]
    bus_p = MMIOBus(READ_ADDR, WRITE_ADDR, [0xABCD, 0x1234])
    bus_s = MMIOBus(READ_ADDR, WRITE_ADDR, [0xABCD, 0x1234])
    res = run(prog, MEM, mmio=bus_p)
    seq = run_sequential(prog, MEM, mmio=bus_s)
    _show_commit_log(res)
    q, rlog, wlog = bus_p.snapshot()
    print(f"设备状态：剩余队列={list(q)} 读日志={list(rlog)} 写记录={list(wlog)}")
    print(f"寄存器：r2={res.regs[2]:#x} r3={res.regs[3]:#x}")
    _check_equiv(res, seq, bus_p, bus_s, "case1")

    _bar("2. 分支冲刷：错误路径上的设备读/写不得发生")
    prog = [
        Instruction.addi(1, 0, READ_ADDR),
        Instruction.beq(0, 0, 3),            # 恒跳转，跳过错误路径
        Instruction.load(2, 1, 0),           # 被冲刷：设备读
        Instruction.store(2, 1, 4),          # 被冲刷：设备写
        Instruction.addi(5, 0, 9),
        Instruction.halt(),
    ]
    bus_p = MMIOBus(READ_ADDR, WRITE_ADDR, [77])
    bus_s = MMIOBus(READ_ADDR, WRITE_ADDR, [77])
    res = run(prog, MEM, mmio=bus_p)
    seq = run_sequential(prog, MEM, mmio=bus_s)
    _show_commit_log(res)
    q, rlog, wlog = bus_p.snapshot()
    print(f"设备状态：剩余队列={list(q)} 读日志={list(rlog)} 写记录={list(wlog)}"
          "（队首 77 未被弹出，写记录为空）")
    _check_equiv(res, seq, bus_p, bus_s, "case2")

    _bar("3. 设备空读：精确异常 MMIO_EMPTY，年轻 MMIO 写被丢弃")
    prog = [
        Instruction.addi(1, 0, READ_ADDR),
        Instruction.addi(5, 0, 8),
        Instruction.store(5, 0, 0),          # 老指令：必须先落内存
        Instruction.load(2, 1, 0),           # 空队列读 -> Fault.MMIO_EMPTY
        Instruction.store(5, 1, 4),          # 年轻 MMIO 写：不得发生
        Instruction.halt(),
    ]
    bus_p = MMIOBus(READ_ADDR, WRITE_ADDR, [])
    bus_s = MMIOBus(READ_ADDR, WRITE_ADDR, [])
    res = run(prog, MEM, mmio=bus_p)
    seq = run_sequential(prog, MEM, mmio=bus_s)
    _show_commit_log(res)
    print(f"故障：{res.fault.value}  pc={res.fault_pc}  cycle={res.fault_cycle}")
    print(f"异常前内存 mem[0]={int.from_bytes(res.memory[0:4], 'little')}"
          f"（老 store 已提交）；设备写记录={list(bus_p.snapshot()[2])}（为空）")
    assert res.fault is Fault.MMIO_EMPTY
    _check_equiv(res, seq, bus_p, bus_s, "case3")

    _bar("演示完成：三种场景均与顺序参考逐条一致")


if __name__ == "__main__":
    main()

