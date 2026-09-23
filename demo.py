"""tinycpu 固定输入演示（确定性，无网络/无 sleep/无预录结果）。

运行：``python demo.py``

内容：
1. 正常程序：含 load-use 停顿气泡、连续同寄存器前递、分支冲刷；
   打印每周期流水槽、退休轨迹，并与独立顺序参考解释器核对；
2. 异常边界：分支冲刷后紧跟非法 STORE，验证"非法访存到 WB 才报告、
   年轻 STORE 不写内存、老指令正常提交"；
3. 输入拒绝：bool 冒充寄存器号被可定位地拒绝。
"""

from __future__ import annotations

from tinycpu import (
    Instruction,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run
from tinycpu.ooo import REORDER_WINDOW, run_out_of_order
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
        if slot.reason == "hold":
            return f" hold{slot.index:<3}"
        if slot.reason == "fault":
            return f" fault{slot.index:<2}"
        if slot.reason == "take":
            return f" take{slot.index:<3}"
        return "   ·    "
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
        "      holdN=停顿冻结   ~flushN=分支冲刷受害者N   faultN=故障丢弃N   takeN=分支命中"
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


def _print_window(result) -> None:
    print(f"周期 | {'重排序窗口（左=最老，右=最年轻；4 项）':<44} 取指→ | 事件")
    for w in result.window:
        cells = []
        for i in range(REORDER_WINDOW):
            if i < len(w.slots):
                s = w.slots[i]
                mark = "✓" if s.state == "done" else "·"
                cells.append(f"{mark}pc{s.pc:<3}")
            else:
                cells.append("     ")
        event = ""
        if w.squashed:
            event = f"冲刷回收 pc{list(w.squashed)} -> pc{w.redirect}"
        print(f"{w.cycle:^4} | {' '.join(cells)}  {w.fetch_pc:<4} | {event}")
    print("\n图例：✓=已完成待退休  · =执行中/等待操作数；取指→=下一条待取指令下标")


def demo_ooo_normal() -> None:
    _bar("演示 3（乱序核心·正常）：算术乱序完成、按序退休、分支回收年轻结果")
    # 0 LOAD r1,0(r0)   mem[0]=5；c1 访存相位才完成
    # 1 BEQ r1,r1,+4    等 LOAD，c2 才解析，命中 -> pc5；pc2/pc3 在此前抢跑
    # 2 ADDI r5,r0,99   c1 整数口抢跑完成（推测结果 99）
    # 3 ADDI r6,r0,88   c2 抢跑完成（推测结果 88）；c2 还会顺序补取 pc4
    # 4 ADDI r7,r0,5    c2 刚补取即随分支冲刷（未完成）
    # 5 ADDI r9,r0,7    分支目标（冲刷后重新取指）
    # 6 ADD r4,r1,r9    5+7=12
    # 7 STORE r4,0(r0)  存 12 到 mem[0]
    # 8 HALT
    prog = [
        Instruction.load(1, 0, 0),
        Instruction.beq(1, 1, 4),
        Instruction.addi(5, 0, 99),
        Instruction.addi(6, 0, 88),
        Instruction.addi(7, 0, 5),
        Instruction.addi(9, 0, 7),
        Instruction.add(4, 1, 9),
        Instruction.store(4, 0, 0),
        Instruction.halt(),
    ]
    _print_program(prog)
    ooo = run_out_of_order(prog, bytearray(memory_from_words([5])))
    seq = run_sequential(prog, bytearray(memory_from_words([5])))

    print("\n每周期重排序窗口：")
    _print_window(ooo)

    print("\n乱序完成记录（执行口实际服务顺序，可能与程序序不同）：")
    for c in ooo.completions:
        tail = "（被分支回收，不退休）" if c.squashed else ""
        extra = ""
        if c.kind == "alu" or c.kind == "load":
            extra = f"值={c.value}"
        elif c.kind == "branch":
            extra = "命中" if c.taken else "不命中"
        elif c.kind == "store":
            extra = f"地址={c.addr} 值={c.value}"
        print(f"  周期{c.cycle:>2}  pc{c.pc:<2} {c.op.name:<5} {extra}{tail}")

    print("\n退休轨迹（严格按程序序）：")
    for t in ooo.trace:
        if t.rd is not None:
            status = f"r{t.rd} <- {t.value}"
        elif t.addr is not None:
            status = f"mem[{t.addr}] <- {t.mem_value}"
        elif t.op.name == "HALT":
            status = "停机"
        else:
            status = "分支/无写"
        print(f"  周期{t.cycle:>2}  pc{t.pc:<2} {t.op.name:<5} {status}")

    print(f"\n最终：r1={ooo.regs[1]} r4={ooo.regs[4]} r9={ooo.regs[9]} "
          f"mem={words_from_memory(ooo.memory)}（期望 5/12/7/[12]）")
    print(f"被回收的年轻结果：r5={ooo.regs[5]} r6={ooo.regs[6]} r7={ooo.regs[7]}"
          f"（期望全 0，其中 99/88 曾真实算出后随标签回收）")
    print(f"总周期 {ooo.cycles_used}；退休序列 {[t.pc for t in ooo.trace]}")
    assert ooo.regs == seq.regs and ooo.memory == seq.memory
    assert [t.pc for t in ooo.trace] == [t.pc for t in seq.trace]
    print("与独立顺序参考解释器逐退休核对：寄存器/内存/退休序列完全一致 ✔")


def demo_ooo_fault() -> None:
    _bar("演示 4（乱序核心·异常边界）：非法访存到退休才报告，年轻结果全丢弃")
    # 0 ADDI r1,r0,3   非对齐基址
    # 1 LOAD r2,0(r1)  非法（执行期识别，退休期报告）
    # 2 ADDI r3,r0,42  与故障无关：会先乱序完成，但故障时随窗口回收
    # 3 STORE r3,0(r0) 年轻 STORE：顺序访存使其根本没执行
    # 4 HALT
    prog = [
        Instruction.addi(1, 0, 3),
        Instruction.load(2, 1, 0),
        Instruction.addi(3, 0, 42),
        Instruction.store(3, 0, 0),
        Instruction.halt(),
    ]
    _print_program(prog)
    mem = memory_from_words([1234])
    ooo = run_out_of_order(prog, bytearray(mem))
    seq = run_sequential(prog, bytearray(mem))

    print("\n乱序完成记录：")
    for c in ooo.completions:
        print(f"  周期{c.cycle:>2}  pc{c.pc:<2} {c.op.name:<5} "
              f"{'地址非法(不触碰内存)' if c.bad else (f'值={c.value}')}")
    print(f"\n故障 {ooo.fault.value}：pc{ooo.fault_pc}，报告于退休周期 "
          f"{ooo.fault_cycle}；退休序列 {[t.pc for t in ooo.trace]}")
    print(f"内存保持 {words_from_memory(ooo.memory)}（年轻 STORE 未写）；"
          f"r2={ooo.regs[2]} r3={ooo.regs[3]}（乱序结果未退休，全 0）")
    assert ooo.fault_pc == seq.fault_pc and ooo.memory == bytes(mem)
    print("与独立顺序参考解释器核对：故障位置与异常前内存一致 ✔")


def demo_ooo_bad_window() -> None:
    _bar("演示 5（乱序核心·输入拒绝）：window 参数非法时无半更新")
    mem = bytearray(memory_from_words([0]))
    snapshot = bytes(mem)
    for bad in (0, 99, True):
        try:
            run_out_of_order([Instruction.halt()], mem, window=bad)
        except Exception as exc:
            print(f"window={bad!r:<5} -> {type(exc).__name__}: {exc}")
    assert bytes(mem) == snapshot
    print("错误均在第一个周期之前抛出；调用方内存未被修改。")


def demo_input_rejection() -> None:
    _bar("演示 6（输入拒绝）：bool 冒充整数 / NaN 立即数，可定位错误")

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


def main() -> None:
    demo_normal()
    print()
    demo_precise_fault()
    print()
    demo_ooo_normal()
    print()
    demo_ooo_fault()
    print()
    demo_ooo_bad_window()
    print()
    demo_input_rejection()
    print("=" * WIDTH)
    print("演示结束：全部为真实计算，结果由 tinycpu 顺序/乱序核心当场产生。")


if __name__ == "__main__":
    main()
