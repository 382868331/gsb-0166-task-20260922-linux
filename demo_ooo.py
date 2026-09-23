"""tinycpu 乱序核心（4 项 ROB + 寄存器重命名）固定输入演示。

运行：``python demo_ooo.py``

四个部分（全部当场计算，无 sleep、无网络、无预录答案）：
1. 正常：双串行 LOAD 堵门，年轻独立 ALU 先于老 LOAD 乱序完成；依赖装入
   值的 BEQ 命中，已完成的年轻算术结果被回收（含 ROB 满前端停顿、重命名
   标签）；逐周期打印派发/执行/完成/退休/回收，并与独立顺序参考逐条核对；
2. 边界·访存保序：年轻 LOAD 必须等老 STORE 退休后才启动，读回存入的值；
3. 实际错误：非法 LOAD 延迟到 ROB 头部退休才报告，已乱序完成的年轻 ALU
   被丢弃，老 STORE 正常提交（精确状态）；
4. 输入拒绝：bool/非对齐内存/无 HALT 在第一个周期前被拒绝，无部分状态。
"""

from __future__ import annotations

from tinycpu import (
    Instruction,
    memory_from_words,
    words_from_memory,
)
from tinycpu.ooo import ROB_SIZE, run_ooo
from tinycpu.sequential import run_sequential

WIDTH = 80


def _bar(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("-" * WIDTH)


def _print_program(prog) -> None:
    for i, ins in enumerate(prog):
        print(f"  pc{i:<2} {ins}")


def _fmt_list(xs):
    return ",".join(str(x) for x in xs) if xs else "·"


def _print_cycles(r) -> None:
    print("周期 | 派发  执行     完成      退休 | 事件")
    for c in r.cycles:
        d = f"pc{c.dispatch.pc}" if c.dispatch else "  · "
        e = _fmt_list(c.execute)
        cp = _fmt_list(c.complete)
        rt = f"pc{c.retire}" if c.retire is not None else " · "
        events = []
        if c.dispatch_stall:
            events.append(f"前端停顿({c.dispatch_stall},窗口{ROB_SIZE}项已满)")
        if c.squash:
            sq = c.squash
            events.append(
                f"分支 pc{sq.branch_pc} 命中->pc{sq.target}，"
                f"回收年轻 {sq.squashed}，其中已完成被丢弃 {sq.reclaimed_completed}"
            )
        if c.fault_discard:
            events.append(f"故障报告，丢弃窗口内年轻指令 {c.fault_discard}")
        ev = "；".join(events) if events else ""
        print(f" {c.cycle:>2}  | {d:>4}  {e:>5}  {cp:>7}  {rt:>4} | {ev}")
    print(
        "\n图例：执行=本周期开始(ALU/分支当拍完成；LOAD 开始访存)，"
        "完成=结果就绪；· = 无事件。"
    )


def _print_retire(r) -> None:
    print("退休轨迹（严格按程序序）：")
    for t in r.trace:
        if t.fault:
            extra = f"故障 {t.fault.value}（不提交）"
        elif t.rd is not None:
            extra = f"r{t.rd} <- {t.value}"
        elif t.addr is not None:
            extra = f"mem[{t.addr}] <- {t.mem_value}"
        elif t.op.name == "HALT":
            extra = "停机"
        else:
            extra = "（无架构写）"
        print(f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {extra}")


def _check_seq(prog, mem, init, label):
    seq = run_sequential(prog, bytearray(mem), init)
    ooo = run_ooo(prog, bytearray(mem), init)
    same = (
        ooo.regs == seq.regs
        and ooo.memory == seq.memory
        and ooo.fault == seq.fault
        and ooo.fault_pc == seq.fault_pc
        and ooo.halted == seq.halted
        and [(t.pc, t.op, t.fault, t.rd, t.value, t.addr, t.mem_value)
             for t in ooo.trace]
        == [(t.pc, t.op, t.fault, t.rd, t.value, t.addr, t.mem_value)
            for t in seq.trace]
    )
    assert same, f"{label}: 乱序核心与顺序参考不一致"
    print(f"与独立顺序参考核对：寄存器/内存/退休序列/终止方式完全一致 ✔")
    return ooo, seq


def demo_ooo_normal() -> None:
    _bar("演示 1（正常）：乱序完成 + ROB 满停顿 + 分支命中回收年轻结果")
    prog = [
        Instruction.addi(8, 0, 0),        # pc0 r8 = 0（安全基址）
        Instruction.load(9, 8, 0),        # pc1 r9 = mem[0] = 7（堵门 LOAD）
        Instruction.load(10, 8, 0),       # pc2 r10 = mem[0] = 7（保序，等 pc1）
        Instruction.beq(10, 5, 4),        # pc3 7==r5(7) 命中 -> pc7
        Instruction.addi(11, 0, 55),      # pc4 独立 ALU，抢先完成，后被回收
        Instruction.addi(12, 0, 66),      # pc5 未完成即被回收
        Instruction.addi(13, 0, 77),      # pc6 窗口满，根本未派发
        Instruction.addi(14, 0, 7),       # pc7 分支目标
        Instruction.halt(),               # pc8
    ]
    _print_program(prog)
    init = [0] * 32
    init[5] = 7
    mem = memory_from_words([7, 0, 0, 0])

    r, _ = _check_seq(prog, mem, init, "demo1")
    print()
    _print_cycles(r)

    print()
    _print_retire(r)
    print(
        f"\n最终寄存器：r9={r.regs[9]} r10={r.regs[10]} r14={r.regs[14]}"
        f"（期望 7/7/7）；r11={r.regs[11]} r12={r.regs[12]} r13={r.regs[13]}"
        f"（被回收/未派发，期望全 0）"
    )
    print(f"退休下标：{[t.pc for t in r.trace]}（跳过年轻 pc4/5/6）")
    # 找两条关键时序证据
    c_alu = next(c.cycle for c in r.cycles if 4 in c.complete)
    c_load2 = next(c.cycle for c in r.cycles if 2 in c.complete)
    print(f"乱序完成证据：年轻 pc4 在周期 {c_alu} 完成，"
          f"早于更老 LOAD pc2 的完成周期 {c_load2}")
    sq = r.squashes[0]
    print(f"回收证据：周期 {sq.cycle} 分支命中，丢弃 {sq.squashed}，"
          f"其中已完成的 {sq.reclaimed_completed} 结果不落架构状态")


def demo_memory_order() -> None:
    _bar("演示 2（边界·访存保序）：年轻 LOAD 等老 STORE 退休后才执行")
    prog = [
        Instruction.addi(1, 0, 0),                  # pc0 r1 = 0
        Instruction.addi(2, 0, 0x41424344),         # pc1 r2 = 0x41424344
        Instruction.store(2, 1, 0),                 # pc2 mem[0] <- r2
        Instruction.load(3, 1, 0),                  # pc3 读同一字
        Instruction.add(4, 3, 2),                   # pc4 r4 = r3 + r2
        Instruction.halt(),                         # pc5
    ]
    _print_program(prog)
    mem = memory_from_words([0, 0])

    r, _ = _check_seq(prog, mem, None, "demo2")
    print()
    _print_cycles(r)
    print()
    _print_retire(r)
    ex_load = next(c.cycle for c in r.cycles if 3 in c.execute)
    rt_store = next(t.cycle for t in r.trace if t.pc == 2)
    print(
        f"\n内存：{[hex(x) for x in words_from_memory(r.memory)]}；"
        f"r3={hex(r.regs[3])}（读回老 STORE 写入值，不是旧内存 0），"
        f"r4={hex(r.regs[4])}"
    )
    print(f"保序证据：老 STORE pc2 退休于周期 {rt_store}，"
          f"年轻 LOAD pc3 最早周期 {ex_load} 才启动（不早于退休）")


def demo_precise_fault() -> None:
    _bar("演示 3（实际错误）：非法 LOAD 在头部退休才报告，年轻结果丢弃")
    prog = [
        Instruction.addi(2, 0, 123),     # pc0
        Instruction.addi(1, 0, 0),       # pc1
        Instruction.store(2, 1, 0),      # pc2 合法老 STORE -> mem[0]=123
        Instruction.addi(3, 0, 2),       # pc3 非对齐基址 r3=2
        Instruction.load(5, 3, 0),       # pc4 地址 2 非对齐 -> ADDR
        Instruction.addi(6, 0, 9),       # pc5 年轻独立 ALU（可能已完成）
        Instruction.halt(),              # pc6
    ]
    _print_program(prog)
    mem = memory_from_words([0, 0])

    r, seq = _check_seq(prog, mem, None, "demo3")
    print()
    _print_cycles(r)
    print()
    _print_retire(r)
    c5_done = next(c.cycle for c in r.cycles if 5 in c.complete)
    print(
        f"\n故障：{r.fault.value}，指令 pc{r.fault_pc}，报告于周期 "
        f"{r.fault_cycle}（执行期只标记，延迟到 ROB 头部退休）"
    )
    print(f"异常前内存：{words_from_memory(r.memory)}（老 STORE 提交 123）")
    print(f"r6={r.regs[6]}（年轻 pc5 虽在周期 {c5_done} 乱序完成，"
          f"故障时被丢弃，不落架构寄存器）；HALT 未退休：{not r.halted}")


def demo_input_rejection() -> None:
    _bar("演示 4（输入拒绝）：错误在第一个周期前抛出，无半更新状态")
    try:
        Instruction.add(True, 1, 2)  # noqa: FBT003 故意 bool
    except TypeError as exc:
        print(f"Instruction.add(True,1,2) -> TypeError: {exc}")

    try:
        run_ooo([Instruction.addi(1, 0, 1)], memory_from_words([0]))
    except Exception as exc:
        print(f"程序不以 HALT 结尾 -> {type(exc).__name__}: {exc}")

    try:
        run_ooo([Instruction.halt()], b"abc")
    except Exception as exc:
        print(f"内存长度非 4 的倍数 -> {type(exc).__name__}: {exc}")

    caller_mem = bytearray(b"\x01\x02\x03\x04")
    snapshot = bytes(caller_mem)
    try:
        run_ooo([Instruction.addi(1, 0, float("nan"))], caller_mem)
    except Exception as exc:
        print(f"NaN 立即数 -> {type(exc).__name__}: {exc}")
    assert bytes(caller_mem) == snapshot
    print("调用方内存在失败后保持原样 ✔；以上错误均在任何周期执行之前抛出。")


def main() -> None:
    demo_ooo_normal()
    print()
    demo_memory_order()
    print()
    demo_precise_fault()
    print()
    demo_input_rejection()
    print("=" * WIDTH)
    print("演示结束：结果全部由 4 项 ROB + 重命名乱序核心当场产生，"
          "并由独立顺序参考逐条核对。")


if __name__ == "__main__":
    main()
