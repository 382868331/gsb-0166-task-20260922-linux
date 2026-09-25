"""tinycpu 固定输入演示（确定性，无网络/无 sleep/无预录结果）。

运行：``python demo.py``

内容：
1. 正常程序：含 load-use 停顿气泡、连续同寄存器前递、分支冲刷；
   打印每周期流水槽、退休轨迹，并与独立顺序参考解释器核对；
2. 异常边界：分支冲刷后紧跟非法 STORE，验证"非法访存到 WB 才报告、
   年轻 STORE 不写内存、老指令正常提交"；
3. 输入拒绝：bool 冒充寄存器号被可定位地拒绝；
4. 中断正常路径：退休边界接纳、保存 PC、冲刷年轻 STORE（处理器读到
   旧内存值）、IRET 返回后重执行；
5. 中断边界：异常与中断同拍——异常优先，中断保持 pending。
"""

from __future__ import annotations

from tinycpu import (
    Instruction,
    InterruptController,
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
        if slot.reason == "irq":
            if slot.index is not None:
                return f" ~irq{slot.index:<3}"
            return "  ~irq  "
        if slot.reason == "hold":
            return f" hold{slot.index:<3}"
        if slot.reason == "fault":
            return f" fault{slot.index:<2}"
        if slot.reason == "take":
            return f" take{slot.index:<3}"
        return "   ·    "
    if slot.reason == "iret":
        return f" iret{slot.index:<3}"
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
        "      holdN=停顿冻结   ~flushN=分支冲刷受害者N   faultN=故障丢弃N   takeN=分支命中\n"
        "      ~irqN=中断冲刷受害者N   iretN=IRET 改向"
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


def demo_input_rejection() -> None:
    _bar("演示 3（输入拒绝）：bool 冒充整数 / NaN 立即数，可定位错误")

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
    print("以上错误均在任何周期执行之前抛出，无部分状态。")


def demo_interrupt() -> None:
    _bar("演示 4（中断正常路径）：退休边界接纳、保存 PC、冲刷年轻 STORE、IRET 返回")
    # 布局：pc0 跳过处理器；处理器 pc1-pc2；主程序 pc3 起。
    prog = [
        Instruction.beq(0, 0, 3),       # pc0  跳过处理器 -> pc3
        Instruction.load(6, 0, 0),      # pc1  处理器：读被冲刷 STORE 的地址（向量 pc=1）
        Instruction.iret(),             # pc2  恢复保存的 PC，随后解除屏蔽
        Instruction.addi(5, 0, 111),    # pc3  r5 = 111
        Instruction.store(5, 0, 0),     # pc4  接纳时在 MEM 级，被冲刷，不得写！
        Instruction.store(6, 0, 4),     # pc5  把处理器读到的值存到 mem[4]
        Instruction.halt(),             # pc6
    ]
    _print_program(prog)

    mem = memory_from_words([0, 0])
    # 离线控制器：在第 1 个退休边界（pc3 退休后）到达一个请求脉冲。
    pipe = run(prog, bytearray(mem),
               interrupts=InterruptController(pulse_boundaries={1}),
               interrupt_vector=1)
    seq = run_sequential(prog, bytearray(mem),
                         interrupts=InterruptController(pulse_boundaries={1}),
                         interrupt_vector=1)

    print("\n每周期流水槽：")
    _print_cycles(pipe)

    ev = pipe.irq_events[0]
    print(f"\n中断事件：第 {ev.boundary} 次退休（周期 {ev.cycle}）后接纳；"
          f"保存 PC={ev.saved_pc}（下一条尚未退休指令 pc4 的 STORE）；转向向量 pc{ev.vector_pc}")
    print(f"退休序列：{[t.pc for t in pipe.trace]}")
    out = words_from_memory(pipe.memory)
    print(f"处理器在 pc1 读到 r6={pipe.regs[6]}（期望 0：被冲刷的 pc4 STORE 没写内存）")
    print(f"最终内存字：{out}（mem[0]=111 是 IRET 返回后 pc4 重新执行写入；mem[4]=0）")
    assert pipe.regs == seq.regs and pipe.memory == seq.memory
    assert [t.pc for t in pipe.trace] == [t.pc for t in seq.trace]
    assert [(e.boundary, e.saved_pc) for e in pipe.irq_events] == \
           [(e.boundary, e.saved_pc) for e in seq.irq_events]
    print("与独立顺序参考（同一边界注入）核对：寄存器 / 内存 / 退休序列 / 中断事件一致 ✔")


def demo_interrupt_fault_priority() -> None:
    _bar("演示 5（实际边界）：异常与中断同拍 —— 异常优先，中断仍待处理")
    prog = [
        Instruction.beq(0, 0, 2),       # pc0  跳过处理器 -> pc2
        Instruction.iret(),             # pc1  处理器（向量 pc=1，本演示不会进入）
        Instruction.addi(1, 0, 3),      # pc2  r1 = 3（非对齐基址）
        Instruction.load(2, 1, 0),      # pc3  非法 LOAD -> WB 报 ADDR
        Instruction.store(2, 0, 0),     # pc4  年轻 STORE，禁止写
        Instruction.halt(),             # pc5
    ]
    _print_program(prog)

    mem = memory_from_words([9, 9])
    # 脉冲恰好落在故障边界（pc3 是第 2 次退休）。
    ctrl = InterruptController(pulse_boundaries={2})
    pipe = run(prog, bytearray(mem), interrupts=ctrl, interrupt_vector=1)
    seq = run_sequential(prog, bytearray(mem),
                         interrupts=InterruptController(pulse_boundaries={2}),
                         interrupt_vector=1)

    print(f"\n故障：{pipe.fault.value}，故障指令 pc{pipe.fault_pc}，报告于周期 {pipe.fault_cycle}")
    print(f"中断接纳事件：{list(pipe.irq_events)}（空 = 异常优先，本边界未接纳）")
    print(f"运行结束后 pending={pipe.irq_pending}（请求保持待处理，未被丢弃）")
    print(f"内存字：{words_from_memory(pipe.memory)}（故障 STORE 与年轻 STORE 均未写）")
    assert pipe.irq_events == () and pipe.irq_pending
    assert pipe.memory == seq.memory and pipe.fault_pc == seq.fault_pc
    assert seq.irq_pending
    print("与独立顺序参考核对：故障位置、异常前内存、pending 状态完全一致 ✔")


def main() -> None:
    demo_normal()
    print()
    demo_precise_fault()
    print()
    demo_input_rejection()
    print()
    demo_interrupt()
    print()
    demo_interrupt_fault_priority()
    print("=" * WIDTH)
    print("演示结束：全部为真实计算，结果由 tinycpu 五级流水核心当场产生。")


if __name__ == "__main__":
    main()
