"""tinycpu 固定输入演示（确定性，无网络/无 sleep/无预录结果）。

运行：``python demo.py``

内容：
1. 正常程序：含 load-use 停顿气泡、连续同寄存器前递、分支冲刷；
   打印每周期流水槽、退休轨迹，并与独立顺序参考解释器核对；
2. 异常边界：分支冲刷后紧跟非法 STORE，验证"非法访存到 WB 才报告、
   年轻 STORE 不写内存、老指令正常提交"；
3. 中断正常路径：在一条 main 指令正常退休的边界注入脉冲，打印
   "接纳 -> 保存 PC -> 冲刷年轻指令 -> 固定向量处理例程 -> IRET 恢复"
   的完整流水槽与退休轨迹，并与顺序参考在同一退休边界注入后比较；
4. 异常与中断同拍：故障指令 WB 报告的同一拍给脉冲，异常优先、
   中断保持 pending；
5. 输入拒绝：bool 冒充寄存器号、非法中断向量被可定位地拒绝。
"""

from __future__ import annotations

from tinycpu import (
    BoundaryInterrupts,
    Instruction,
    InterruptController,
    VECTOR_PC,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

WIDTH = 78


def _bar(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("-" * WIDTH)


def _cell(slot) -> str:
    """一个流水槽的紧凑文本（固定 8 字符宽）：iN 指令下标；气泡用符号+原因。"""
    if slot.reason == "fault":
        # WB 槽是故障报告者（非气泡），其余级是被丢弃的年轻指令。
        label = "!" if not slot.bubble else "~"
        return f" {label}fault{slot.index if slot.index is not None else '·'}"[:8].ljust(8)
    if slot.bubble:
        if slot.reason == "stall":
            return " ~stall "
        if slot.reason == "flush":
            if slot.index is not None:
                return f" ~flush{slot.index}"
            return " ~flush "
        if slot.reason == "intr":
            return f" ~intr{slot.index if slot.index is not None else '·'}"
        if slot.reason == "iret":
            return f" ~iret{slot.index if slot.index is not None else '·'}"
        if slot.reason == "hold":
            return f" hold{slot.index:<3}"
        if slot.reason == "fault":
            return f" fault{slot.index:<2}"
        if slot.reason == "take":
            return f" take{slot.index:<3}"
        return "   ·    "
    if slot.reason == "take":
        return f" take{slot.index:<3}"
    return f" i{slot.index:<6}"


def _print_cycles(result, limit: int | None = None) -> None:
    print("周期 |  IF     ID     EX     MEM    WB")
    cycles = result.cycles if limit is None else result.cycles[:limit]
    for c in cycles:
        cells = "".join(_cell(s) for s in c.slots)
        print(f"{c.cycle:^4} |{cells}")
    if limit is not None and len(result.cycles) > limit:
        print(f"...（共 {len(result.cycles)} 周期，以下省略）...")
    print(
        "\n图例：iN=指令N   · =启动/排空气泡   ~stall=load-use 气泡\n"
        "      holdN=停顿冻结   ~flushN=分支冲刷受害者N   faultN=故障丢弃N\n"
        "      takeN=分支命中   ~intrN=中断接纳冲刷N   ~iretN=IRET 冲刷N"
    )


def _print_program(prog) -> None:
    for i, ins in enumerate(prog):
        print(f"  pc{i:<2} {ins}")


def demo_normal() -> None:
    _bar("演示 1（正常结果）：load-use 停顿 + 同寄存器依赖链 + 分支冲刷")
    prog = [
        Instruction.addi(1, 0, 40),     # pc0  r1 = 40（合法对齐基址）
        Instruction.addi(2, 0, 3),      # pc1  r2 = 3
        Instruction.addi(3, 0, 3),      # pc2  r3 = 3
        Instruction.load(4, 1, 0),      # pc3  r4 = mem[40]
        Instruction.add(5, 4, 2),       # pc4  r5 = r4 + r2（load-use -> 停 1 拍）
        Instruction.addi(5, 5, 1),      # pc5  r5 += 1（同寄存器，EX 前递）
        Instruction.beq(2, 3, 3),       # pc6  3==3 命中 -> pc9，冲刷 pc7/pc8
        Instruction.store(5, 1, 4),     # pc7  （被冲刷）
        Instruction.sub(6, 5, 2),       # pc8  （被冲刷）
        Instruction.addi(6, 0, 100),    # pc9  r6 = 100（分支目标）
        Instruction.store(5, 1, 8),     # pc10 mem[48] = r5
        Instruction.halt(),             # pc11
    ]
    _print_program(prog)

    words = [0] * 13
    words[10] = 7          # 字节地址 40 处放 7
    mem = memory_from_words(words)

    pipe = run(prog, bytearray(mem))
    seq = run_sequential(prog, bytearray(mem))   # 独立参考，不调用核心

    print("\n每周期流水槽：")
    _print_cycles(pipe)

    print("\n退休轨迹（按 WB 顺序）：")
    for t in pipe.trace:
        extra = ""
        if t.rd is not None:
            extra = f"r{t.rd} <- {t.value}"
        elif t.addr is not None:
            extra = f"mem[{t.addr}] <- {t.mem_value}"
        elif t.op.name == "HALT":
            extra = "停机"
        print(f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {extra}")

    out_words = words_from_memory(pipe.memory)
    print(f"\n最终寄存器：r1={pipe.regs[1]} r2={pipe.regs[2]} r4={pipe.regs[4]} "
          f"r5={pipe.regs[5]} r6={pipe.regs[6]}（期望 40/3/7/11/100）")
    print(f"内存字：mem[40]={out_words[10]}（装入值，未改写），"
          f"mem[48]={out_words[12]}（pc10 存入，期望 11）")
    print(f"退休指令下标：{[t.pc for t in pipe.trace]}（pc7/pc8 被冲刷，不出现）")
    print(f"总周期数：{pipe.cycles_used}；HALT 退休：{pipe.halted}；异常：{pipe.fault}")
    assert pipe.regs == seq.regs and pipe.memory == seq.memory
    assert [t.pc for t in pipe.trace] == [t.pc for t in seq.trace]
    print("与独立顺序参考解释器核对：寄存器 / 内存 / 退休序列完全一致 ✔")


def demo_precise_fault() -> None:
    _bar("演示 2（异常边界）：非法访存延迟到 WB 报告，年轻 STORE 不写内存")
    prog = [
        Instruction.addi(1, 0, 5),       # pc0  r1 = 5（非对齐基址）
        Instruction.addi(2, 0, 77),      # pc1  r2 = 77
        Instruction.beq(2, 2, 2),        # pc2  恒真 -> pc4，冲刷 pc3
        Instruction.store(2, 0, 0),      # pc3  被冲刷（否则会写 mem[0]）
        Instruction.store(2, 1, 0),      # pc4  地址 5 非对齐 -> WB 报 ADDR
        Instruction.store(2, 0, 4),      # pc5  年轻且地址合法，但禁止写！
        Instruction.halt(),              # pc6
    ]
    _print_program(prog)

    mem = memory_from_words([999, 888, 777])
    pipe = run(prog, bytearray(mem))
    seq = run_sequential(prog, bytearray(mem))

    print("\n故障周期（最后一个周期）流水槽：")
    last = pipe.cycles[-1]
    print("      " + "".join(f"{s.stage:^8}" for s in last.slots))
    print("     |" + "".join(_cell(s) for s in last.slots))
    print("\n退休轨迹：")
    for t in pipe.trace:
        if t.fault:
            status = f"故障 {t.fault.value}（不提交）"
        elif t.rd is not None:
            status = f"r{t.rd} <- {t.value}"
        elif t.addr is not None:
            status = f"mem[{t.addr}] <- {t.mem_value}"
        elif t.op.name == "HALT":
            status = "停机"
        else:
            status = "（无架构写）"
        print(f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {status}")

    print(f"\n故障：{pipe.fault.value if pipe.fault else None}，"
          f"故障指令 pc{pipe.fault_pc}，报告于周期 {pipe.fault_cycle}（EX 已先判错，延迟到 WB）")
    print(f"异常前/后内存：{words_from_memory(pipe.memory)}（保持 999/888/777）")
    print(f"  - pc3 被分支冲刷，未写 mem[0]；")
    print(f"  - pc4 非法 STORE 自身不写；pc5 更年轻，故障时尚在 MEM，未到 WB，不写；")
    print(f"  - pc0/pc1/pc2 等更老指令均已正常退休。")
    assert pipe.memory == bytes(mem)
    assert pipe.fault_pc == seq.fault_pc and pipe.memory == seq.memory
    print("与独立顺序参考解释器核对：故障位置与异常前内存完全一致 ✔")


def _trace_line(t) -> str:
    extra = ""
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
    if t.event == "INTR":
        extra += f"  ◀ 接纳中断：保存 pc{t.saved_pc}，转向量 pc{VECTOR_PC}"
    elif t.event == "IRET":
        extra += f"  ◀ IRET：恢复 pc{t.saved_pc}，解除屏蔽"
    return f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {extra}"


def demo_interrupt_normal() -> None:
    _bar("演示 3（中断正常路径）：退休边界接纳、保存 PC、冲刷、IRET 恢复")
    # 固定向量 VECTOR_PC=0。冷启动也从 pc0 起，用 r30 区分两种进入：
    #   冷启动 r30=1 走 pc1 跳过例程；中断进入时 r30=0 落入 pc2 例程。
    prog = [
        Instruction.beq(30, 0, 2),       # pc0 中断入口 -> pc2
        Instruction.beq(0, 0, 3),        # pc1 冷启动 -> pc4
        Instruction.addi(20, 20, 1),     # pc2 处理例程：r20 += 1
        Instruction.iret(),              # pc3 IRET：恢复保存 PC
        Instruction.addi(30, 0, 0),      # pc4 main：清冷启动标志
        Instruction.addi(1, 0, 40),      # pc5 基址
        Instruction.addi(2, 0, 3),       # pc6
        Instruction.load(4, 1, 0),       # pc7 r4 = mem[40]
        Instruction.add(5, 4, 2),        # pc8 load-use（停一拍）
        Instruction.store(5, 1, 8),      # pc9 mem[48] = r5
        Instruction.halt(),              # pc10
    ]
    _print_program(prog)
    words = [0] * 13
    words[10] = 7
    mem = memory_from_words(words)
    init = [0] * 32
    init[30] = 1

    # 无中断干跑：确定 pc5 退休周期与退休边界编号，两个执行器在该同一
    # 退休边界注入脉冲（流水线按周期、顺序机按已退休指令数）。
    dry = run(prog, bytearray(mem), init)
    accept_cycle = next(t.cycle for t in dry.trace if t.pc == 5)
    pcs = [t.pc for t in dry.trace]
    accept_boundary = pcs.index(5)

    ic = InterruptController().request(accept_cycle)
    bc = BoundaryInterrupts().request_after(accept_boundary)
    pipe = run(prog, bytearray(mem), init, ic)
    seq = run_sequential(prog, bytearray(mem), init, bc)

    print(f"\n脉冲注入周期：{accept_cycle}（pc5 正常退休的边界）")
    print("每周期流水槽：")
    _print_cycles(pipe)

    print("\n退休轨迹（含中断接纳 / IRET 事件）：")
    for t in pipe.trace:
        print(_trace_line(t))

    acc = pipe.interrupts[0]
    print(
        f"\n接纳：周期 {acc.cycle}，老指令 pc{acc.retired_pc} 正常退休，"
        f"保存下一条尚未退休指令 pc{acc.saved_pc}；同周期 MEM/EX/ID/IF 的年轻指令"
        "全部冲刷（~intr），下周期从 pc0 取处理例程。"
    )
    print(f"处理例程效果：r20 = {pipe.regs[20]}（期望 1）；main 现场保持："
          f"r4={pipe.regs[4]} r5={pipe.regs[5]}（期望 7/10）")
    print(f"内存 mem[48] = {words_from_memory(pipe.memory)[12]}（期望 10，"
          "恢复后 pc9 才写入；被冲刷的年轻 STORE 不落内存）")
    assert pipe.regs == seq.regs and pipe.memory == seq.memory
    assert [(t.pc, t.event, t.saved_pc) for t in pipe.trace] == \
           [(t.pc, t.event, t.saved_pc) for t in seq.trace]
    print("与独立顺序参考（同退休边界注入）核对：寄存器/内存/退休事件序列一致 ✔")


def demo_interrupt_vs_fault_same_cycle() -> None:
    _bar("演示 4（实际边界）：同步访存异常与中断同一退休边界——异常优先")
    prog = [
        Instruction.addi(1, 0, 2),       # pc0 r1 = 2（非对齐）
        Instruction.load(2, 1, 0),       # pc1 地址 2 非对齐 -> WB 报 ADDR
        Instruction.addi(3, 0, 5),       # pc2 年轻，被丢弃
        Instruction.halt(),              # pc3
    ]
    _print_program(prog)
    mem = memory_from_words([123, 456])
    dry = run(prog, bytearray(mem))
    fault_cycle = dry.fault_cycle

    # 脉冲恰在故障指令 WB 报告那一拍（周期）/同一条指令退休的边界（顺序）。
    ic = InterruptController().request(fault_cycle)
    bc = BoundaryInterrupts().request_after(1)  # pc1 退休前已退休 1 条
    pipe = run(prog, bytearray(mem), interrupts=ic)
    seq = run_sequential(prog, bytearray(mem), interrupts=bc)

    print(f"\n脉冲周期 {fault_cycle} 正是 pc1 在 WB 报 ADDR 的周期。")
    print("退休轨迹：")
    for t in pipe.trace:
        print(_trace_line(t))
    print(
        f"\n结果：fault={pipe.fault.value}（pc{pipe.fault_pc}），"
        f"接纳次数={len(pipe.interrupts)}，中断未被消费 pending={pipe.interrupts_pending}"
    )
    print("机器停在故障点，未转向量 pc0；异常前内存保持："
          f"{words_from_memory(pipe.memory)}（123/456）")
    assert pipe.fault_pc == seq.fault_pc
    assert pipe.interrupts == seq.interrupts == ()
    assert pipe.interrupts_pending and seq.interrupts_pending
    print("顺序参考同边界注入同样异常优先、pending 保持 ✔")


def demo_input_rejection() -> None:
    _bar("演示 5（输入拒绝）：bool 冒充整数 / NaN 立即数 / 非法中断向量")

    try:
        Instruction.add(True, 1, 2)  # noqa: FBT003 - 故意传入 bool
    except TypeError as exc:
        print(f"Instruction.add(True, 1, 2) -> TypeError: {exc}")

    try:
        Instruction.addi(1, 1, float("nan"))
    except ValueError as exc:
        print(f"Instruction.addi(1, 1, nan) -> ValueError: {exc}")

    bad_regs = [0] * 32
    bad_regs[0] = 1
    try:
        run([Instruction.halt()], memory_from_words([0]), bad_regs)
    except Exception as exc:
        print(f"r0 初值非零 -> {type(exc).__name__}: {exc}")

    try:
        InterruptController(vector=7).request(3, vector=7)
    except ValueError as exc:
        print(f"非法中断向量 7 -> ValueError: {exc}")
    print("以上错误均在任何周期执行之前抛出，无部分状态。")


def main() -> None:
    demo_normal()
    print()
    demo_precise_fault()
    print()
    demo_interrupt_normal()
    print()
    demo_interrupt_vs_fault_same_cycle()
    print()
    demo_input_rejection()
    print("=" * WIDTH)
    print("演示结束：全部为真实计算，结果由 tinycpu 五级流水核心当场产生。")


if __name__ == "__main__":
    main()
