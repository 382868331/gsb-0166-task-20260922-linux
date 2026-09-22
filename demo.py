"""固定输入演示：五级流水线的正常运行与精确异常/拒绝边界。

运行：python demo.py
全部数据在本文件中确定性构造，结果即时计算（无 sleep、无预录输出）。
"""

from pipeline_cpu import (
    Instruction,
    Opcode,
    InvalidProgramError,
    run_pipeline,
    run_sequential,
)


def words(data):
    return b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in data)


def _slot(stage):
    if stage.seq is None:
        return "  -- "
    mark = "~" if stage.flushed else ("*" if stage.stalled else " ")
    return f"I{stage.seq:<2}{mark}"


def print_trace(result):
    print(f"{'周期':>4} {'IF':>5} {'ID':>5} {'EX':>5} {'MEM':>5} "
          f"{'WB':>5}  事件")
    for t in result.trace:
        note = {
            "": "",
            "stall": "load-use 停顿（ID/IF 冻结，ID/EX 进气泡）",
            "flush": "分支在 EX 决策，冲刷年轻指令",
            "stall+flush": "停顿与冲刷同周期",
            "halt": "HALT 退休，年轻指令丢弃",
            "fault": "故障到 WB 才报告，年轻阶段冻结",
        }.get(t.note, t.note)
        print(f"{t.cycle:>4} {_slot(t.if_stage):>5} {_slot(t.id_stage):>5} "
              f"{_slot(t.ex_stage):>5} {_slot(t.mem_stage):>5} "
              f"{_slot(t.wb_stage):>5}  {note}")


def print_retirements(result):
    print("  周期  指令  退休动作")
    for r in result.retirements:
        tag = f"I{r.seq}" if r.seq is not None else " ? "
        if r.fault is not None:
            action = f"异常 {r.fault.kind}（地址 {r.fault.address}）"
        elif r.halted:
            action = "HALT 停机"
        elif r.rd is None:
            action = "提交（无写回）"
        elif r.value is None:
            action = f"写 r{r.rd} 的值被丢弃（r0 恒零）"
        else:
            action = f"r{r.rd} = {r.value}"
        print(f"  {r.cycle:>4}  {tag:<4} {r.opcode.value if r.opcode else '-':<5} "
              f"{action}")


def demo_normal():
    print("=" * 72)
    print("场景一：正常循环程序（含 load-use 气泡与分支冲刷）")
    print("=" * 72)
    # mem 字 0 存放常数 10；r1 为计数器，r2=3 为循环上界。
    program = [
        Instruction(Opcode.ADDI, a=1, b=0, imm=0),    # I0  r1 = 0
        Instruction(Opcode.ADDI, a=2, b=0, imm=3),    # I1  r2 = 3
        Instruction(Opcode.ADDI, a=1, b=1, imm=1),    # I2  r1 += 1
        Instruction(Opcode.LOAD, a=3, b=0, imm=0),    # I3  r3 = mem[0]=10
        Instruction(Opcode.ADD, a=4, b=3, c=1),       # I4  r4 = r3+r1 (冒险)
        Instruction(Opcode.STORE, b=0, c=4, imm=4),   # I5  mem[4] = r4
        Instruction(Opcode.BEQ, b=1, c=2, imm=8),     # I6  相等 -> I8 HALT
        Instruction(Opcode.BEQ, b=0, c=0, imm=-20),   # I7  否则 -> I2
        Instruction(Opcode.HALT),                     # I8
    ]
    memory = words([10, 0])

    result = run_pipeline(program, memory)
    ref = run_sequential(program, memory)

    print_trace(result)
    print()
    print("退休轨迹（按 WB 顺序）：")
    print_retirements(result)
    print()
    print(f"总周期数: {result.cycles}（无冒险理想值 = 9+4 = 13，"
          f"额外周期来自 load-use 停顿与 3 次分支冲刷）")
    print(f"最终寄存器 r1..r4 = {result.final_registers[1:5]}")
    print(f"最终内存 mem[0..7] = "
          f"{[int.from_bytes(result.final_memory[i:i + 4], 'little') for i in (0, 4)]}")
    same = (result.final_registers == ref.final_registers
            and result.final_memory == ref.final_memory
            and result.halted == ref.halted)
    print(f"与独立顺序解释器比较寄存器/内存/结局：{'一致 ✓' if same else '不一致 ✗'}")
    stalls = sum("stall" in t.note for t in result.trace)
    flushes = sum("flush" in t.note for t in result.trace)
    print(f"（图例：* 停顿冻结，~ 被冲刷，-- 气泡；共 {stalls} 次停顿，"
          f"{flushes} 次冲刷）")
    return same


def demo_precise_fault():
    print()
    print("=" * 72)
    print("场景二：精确异常边界（非法访存到 WB 才报告，年轻 store 不写）")
    print("=" * 72)
    program = [
        Instruction(Opcode.ADDI, a=1, b=0, imm=201),   # I0 r1 = 201（奇地址）
        Instruction(Opcode.ADDI, a=2, b=0, imm=55),    # I1 r2 = 55
        Instruction(Opcode.STORE, b=0, c=2, imm=0),    # I2 mem[0] = 55（老，合法）
        Instruction(Opcode.STORE, b=1, c=2, imm=0),    # I3 mem[201]：非法
        Instruction(Opcode.STORE, b=0, c=2, imm=8),    # I4 更年轻，禁止提交
        Instruction(Opcode.HALT),                      # I5 更年轻，不退休
    ]
    memory = bytes(16)
    result = run_pipeline(program, memory)
    ref = run_sequential(program, memory)

    print("故障前后的流水槽（最后两个周期）：")
    print(f"{'周期':>4} {'IF':>5} {'ID':>5} {'EX':>5} {'MEM':>5} {'WB':>5}")
    for t in result.trace[-2:]:
        print(f"{t.cycle:>4} {_slot(t.if_stage):>5} {_slot(t.id_stage):>5} "
              f"{_slot(t.ex_stage):>5} {_slot(t.mem_stage):>5} "
              f"{_slot(t.wb_stage):>5}  "
              + {"fault": "I3 在 WB 报告；I4 停在 MEM 被冻结丢弃"}.get(
                  t.note, ""))
    print()
    print(f"报告结果：fault={result.fault.kind}，地址 {result.fault.address}，"
          f"故障指令 I{result.fault_seq}，退休于周期 {result.fault_cycle}")
    mem_words = [int.from_bytes(result.final_memory[i:i + 4], "little")
                 for i in range(0, 16, 4)]
    print(f"异常前内存（老指令 I2 已提交，I3/I4 均未写）：{mem_words}")
    print(f"退休指令序号：{[r.seq for r in result.retirements]}（I4/I5 不在其中）")
    same = (result.final_memory == ref.final_memory
            and result.final_registers == ref.final_registers
            and result.fault.address == ref.fault.address)
    print(f"与独立顺序解释器的“异常前快照”比较：{'一致 ✓' if same else '不一致 ✗'}")
    return same


def demo_rejected_input():
    print()
    print("=" * 72)
    print("场景三：非法输入拒绝边界（可定位错误，不静默纠正）")
    print("=" * 72)
    ok = True
    try:
        run_pipeline(
            [Instruction(Opcode.ADDI, a=1, b=0, imm=float("nan")),
             Instruction(Opcode.HALT)],
            bytes(16),
        )
        ok = False
        print("错误：NaN 立即数未被拒绝！")
    except InvalidProgramError as exc:
        print(f"传入 NaN 立即数被拒绝：{type(exc).__name__}: {exc}")

    try:
        run_pipeline(
            [Instruction(Opcode.ADD, a=True, b=0, c=0),  # bool 冒充寄存器号
             Instruction(Opcode.HALT)],
            bytes(16),
        )
        ok = False
        print("错误：bool 寄存器号未被拒绝！")
    except InvalidProgramError as exc:
        print(f"传入 bool 寄存器号被拒绝：{type(exc).__name__}: {exc}")

    try:
        run_pipeline([Instruction(Opcode.HALT)], bytes(15))  # 未对齐内存
        ok = False
        print("错误：未对齐内存未被拒绝！")
    except InvalidProgramError as exc:
        print(f"传入长度 15 字节内存被拒绝：{type(exc).__name__}: {exc}")
    return ok


def main():
    a = demo_normal()
    b = demo_precise_fault()
    c = demo_rejected_input()
    print()
    print("=" * 72)
    if a and b and c:
        print("演示完成：正常结果、精确异常边界、输入拒绝边界均符合契约。")
    else:
        print("演示完成，但存在不一致，请检查上面的 ✗ 标记。")
    print("=" * 72)


if __name__ == "__main__":
    main()
