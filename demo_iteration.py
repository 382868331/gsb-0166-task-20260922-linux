"""分支预测增量的固定输入演示（确定性，无网络/无 sleep/无预录结果）。

运行：python demo_iteration.py

内容：
1. 循环求和程序：接入两位饱和计数器预测器，打印预测次数/错判/冲刷，
   并与独立顺序参考解释器逐条核对退休记录；
2. 别名碰撞：表容量 1 强制两个 BEQ 共用计数器，对比错判次数上升；
3. 错误路径副作用：预测命中跳错路径时，错误路径上的非法 STORE 与
   HALT 不会成为架构结果（内存不变、不报错、机器继续）；
4. 训练时机：被老指令故障冲掉的已解析分支不污染计数器。
"""

from __future__ import annotations

from tinycpu import (
    BranchPredictor,
    Instruction,
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


def _check_vs_sequential(result, seq) -> None:
    """逐条核对提交记录与最终架构状态。"""
    assert len(result.trace) == len(seq.trace), "退休条数不一致"
    for t_pipe, t_seq in zip(result.trace, seq.trace):
        assert (
            t_pipe.pc == t_seq.pc
            and t_pipe.op == t_seq.op
            and t_pipe.fault == t_seq.fault
            and t_pipe.rd == t_seq.rd
            and t_pipe.value == t_seq.value
            and t_pipe.addr == t_seq.addr
            and t_pipe.mem_value == t_seq.mem_value
        ), f"退休记录不一致: {t_pipe} vs {t_seq}"
    assert result.regs == seq.regs, "最终寄存器不一致"
    assert result.memory == seq.memory, "最终内存不一致"
    assert result.fault == seq.fault and result.fault_pc == seq.fault_pc
    print(f"与顺序参考逐条核对一致（{len(result.trace)} 条退休记录）")


def sum_loop_program():
    return [
        Instruction.addi(2, 0, 5),    # 0: r2 = 5（上限）
        Instruction.addi(1, 0, 0),    # 1: r1 = i
        Instruction.addi(3, 0, 0),    # 2: r3 = sum
        Instruction.addi(1, 1, 1),    # 3: i += 1
        Instruction.add(3, 3, 1),     # 4: sum += i
        Instruction.beq(1, 2, 2),     # 5: i == 5 退出到 7
        Instruction.beq(0, 0, -3),    # 6: 否则回 3
        Instruction.halt(),           # 7
    ]


def main() -> None:
    _bar("1. 循环程序 + 两位饱和计数器预测（8 表项，初值 1）")
    prog = sum_loop_program()
    pred = BranchPredictor(entries=8, initial=1)
    result = run(prog, memory_from_words([0] * 4), predictor=pred)
    seq = run_sequential(prog, memory_from_words([0] * 4))
    _check_vs_sequential(result, seq)
    stats = result.prediction
    print(f"sum = {result.regs[3]}（1+2+3+4+5），halted = {result.halted}")
    print(f"预测 {stats.predictions} 次，错判 {stats.mispredictions} 次，"
          f"冲刷 {stats.flushed} 条")
    print(f"训练后计数器：pc5 -> {pred.counter(5)}，pc6 -> {pred.counter(6)}")
    base = run(prog, memory_from_words([0] * 4))
    print(f"对比无预测基线：{base.cycles_used} 周期 -> 带预测 {result.cycles_used} 周期")

    _bar("2. 别名碰撞：表容量 1，两个 BEQ 被迫共用同一计数器")
    pred_alias = BranchPredictor(entries=1, initial=1)
    res_alias = run(prog, memory_from_words([0] * 4), predictor=pred_alias)
    _check_vs_sequential(res_alias, seq)
    s1, s2 = stats, res_alias.prediction
    print(f"8 表项错判 {s1.mispredictions} 次；1 表项（全别名）错判 "
          f"{s2.mispredictions} 次——碰撞降低精度但不影响正确性")

    _bar("3. 错误路径副作用：非法 STORE / HALT 不得成为架构结果")
    prog3 = [
        Instruction.addi(1, 0, 1),      # 0: r1 = 1
        Instruction.addi(2, 0, 2),      # 1: r2 = 2
        Instruction.beq(1, 2, 3),       # 2: 实际不命中；预测命中到 5
        Instruction.store(1, 0, 0),     # 3: 正确路径 mem[0] = 1
        Instruction.beq(0, 0, 2),       # 4: 正确路径跳到 6
        Instruction.store(2, 0, 252),   # 5: 仅错误路径，非法地址
        Instruction.halt(),             # 6
    ]
    pred3 = BranchPredictor(entries=4, initial=3)   # 强命中：走上错误路径
    res3 = run(prog3, memory_from_words([0] * 4), predictor=pred3)
    seq3 = run_sequential(prog3, memory_from_words([0] * 4))
    _check_vs_sequential(res3, seq3)
    print(f"fault = {res3.fault}（错误路径的非法 STORE 未报告），"
          f"halted = {res3.halted}")
    print(f"内存字 = {words_from_memory(res3.memory)}（仅正确路径的 mem[0]=1）")
    print(f"错判 {res3.prediction.mispredictions} 次，"
          f"冲刷 {res3.prediction.flushed} 条")

    _bar("4. 训练时机：被故障冲掉的已解析分支不污染计数器")
    prog4 = [
        Instruction.load(1, 0, 252),    # 0: 非法 LOAD，WB（周期 5）报告
        Instruction.beq(0, 0, 2),       # 1: 命中到 3；周期 4 已在 EX 解析
        Instruction.addi(5, 0, 9),      # 2: 被冲刷
        Instruction.halt(),             # 3
    ]
    pred4 = BranchPredictor(entries=4, initial=1)
    res4 = run(prog4, memory_from_words([0] * 4), predictor=pred4)
    seq4 = run_sequential(prog4, memory_from_words([0] * 4))
    _check_vs_sequential(res4, seq4)
    print(f"fault = {res4.fault.value} @ pc{res4.fault_pc}，"
          f"错判 {res4.prediction.mispredictions} 次（EX 已解析）")
    print(f"pc1 计数器 = {pred4.counter(1)}（仍为初值 1：分支未提交，未训练）")

    print("=" * WIDTH)
    print("demo_iteration 完成：全部核对通过")


if __name__ == "__main__":
    main()

