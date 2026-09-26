"""分支预测增量的真实离线演示（无 sleep、无预录结果、无第三方依赖）。

当场构造三个程序并实跑：
  1. 循环计数：不同表容量/初值下的预测次数、错判与冲刷指令数，
     并逐条对照无预测顺序解释器 tinycpu.run_sequential 的提交记录；
  2. 两个 PC 直接索引别名碰撞：1 项表 vs 8 项独立表，错判率不同但
     架构结果完全一致；
  3. 错误路径对照：预测把"实际跳转"误判为不跳时，错误路径上的
     STORE / HALT / 非法访存都不得变成架构结果（对照顺序解释器）。

运行：python demo_iteration.py   （通常 1 秒内完成，远低于 90 秒预算）
"""

from tinycpu import Instruction, memory_from_words, words_from_memory
from tinycpu.pipeline import BranchPredictor, run
from tinycpu.sequential import run_sequential


def _check_against_sequential(label, prog, mem, init=None):
    seq = run_sequential(prog, bytearray(mem), init)
    return seq


def _print_trace_diff(label, pipe_result, seq_result):
    p = [(t.cycle, t.pc, t.op.value,
          t.rd, t.value, t.addr, t.mem_value, t.fault)
         for t in pipe_result.trace]
    s = [(t.pc, t.op.value, t.rd, t.value, t.addr, t.mem_value, t.fault)
         for t in seq_result.trace]
    sp = [(row[1], row[2], row[3], row[4], row[5], row[6], row[7]) for row in p]
    status = "一致" if sp == s else "不一致"
    print(f"  [{label}] 提交记录逐条对照顺序解释器：{status}"
          f"（流水线 {len(p)} 条，顺序 {len(s)} 条）")
    return sp == s


def main() -> None:
    ok = True

    # ------------------------------------------------------------------
    # 1) 循环：r1 从 4 递减到 0；pc2 = N,N,N,N,T；pc3 = 无条件回跳 3 次。
    # ------------------------------------------------------------------
    print("=== 1) 计数循环上的两位饱和计数器预测 ===")
    loop = [
        Instruction.addi(1, 0, 4),      # 0 r1 = 4
        Instruction.addi(1, 1, -1),     # 1 r1 -= 1
        Instruction.beq(1, 0, 2),       # 2 r1==0 -> 4 退出
        Instruction.beq(0, 0, -2),      # 3 无条件回 1
        Instruction.halt(),             # 4
    ]
    mem = memory_from_words([0])
    seq = _check_against_sequential("loop", loop, mem)
    print(f"  顺序解释器提交序列: {[t.pc for t in seq.trace]}")
    for entries, initial in ((8, 0), (8, 1), (8, 3), (1, 1)):
        bp = BranchPredictor(entries, initial)
        r = run(loop, bytearray(mem), None, bp)
        tag = f"entries={entries}, init={initial}"
        print(f"  {tag:>20}: 预测 {r.predictions} 次, 错判 {r.mispredictions} 次, "
              f"冲刷 {r.flushed_instructions} 条, {r.cycles_used} 周期, "
              f"表={bp.counters if entries <= 8 else '...'}")
        ok &= _print_trace_diff(tag, r, seq)
        assert r.regs == seq.regs and r.memory == seq.memory

    # ------------------------------------------------------------------
    # 2) 两个 PC 碰撞：1 项表里 pc2/pc3 共享同一计数器（反复互相踩），
    #    8 项表中二者独立；架构结果必须相同，错判统计可以不同。
    # ------------------------------------------------------------------
    print("=== 2) 直接索引别名碰撞（pc2 与 pc3 在 1 项表中同槽） ===")
    r_alias = run(loop, bytearray(mem), None, BranchPredictor(1, 1))
    r_split = run(loop, bytearray(mem), None, BranchPredictor(8, 1))
    print(f"  1 项表（别名）: 错判 {r_alias.mispredictions} 次, "
          f"冲刷 {r_alias.flushed_instructions} 条, 终态 r1={r_alias.regs[1]}")
    print(f"  8 项表（独立）: 错判 {r_split.mispredictions} 次, "
          f"冲刷 {r_split.flushed_instructions} 条, 终态 r1={r_split.regs[1]}")
    assert r_alias.regs == r_split.regs == seq.regs
    assert r_alias.mispredictions != r_split.mispredictions
    print("  两表架构结果一致，错判率随别名碰撞而不同：符合预期")

    # ------------------------------------------------------------------
    # 3) 错误路径副作用隔离：实际 taken 被预测成 not-taken。
    #    pc3 错误 STORE（会写 mem[0]）、pc5 错误 HALT、
    #    pc7/pc8 错误路径上构造非法 STORE（基址非对齐）。
    # ------------------------------------------------------------------
    print("=== 3) 错误路径的 store / HALT / 非法访存不得提交 ===")
    prog = [
        Instruction.addi(1, 0, 5),           # 0
        Instruction.addi(2, 0, 5),           # 1
        Instruction.beq(1, 2, 7),            # 2 实际 -> 9；初值 0 预测不跳
        Instruction.addi(6, 0, 3),           # 3 错误路径：非对齐基址
        Instruction.store(1, 0, 0),          # 4 错误路径：合法 store，禁写！
        Instruction.halt(),                  # 5 错误路径 HALT，禁停机！
        Instruction.store(1, 6, 0),          # 6 错误路径非法 STORE，禁报异常
        Instruction.addi(7, 0, 42),          # 7 错误路径
        Instruction.addi(8, 0, 43),          # 8 错误路径
        Instruction.addi(9, 0, 100),         # 9 正确路径
        Instruction.halt(),                  # 10
    ]
    bad_mem = memory_from_words([1234, 0])
    seq2 = run_sequential(prog, bytearray(bad_mem))
    bp = BranchPredictor(16, 0)
    r2 = run(prog, bytearray(bad_mem), None, bp)
    print(f"  预测 {r2.predictions} 次, 错判 {r2.mispredictions} 次, "
          f"冲刷 {r2.flushed_instructions} 条")
    print(f"  halted={r2.halted}, fault={r2.fault}, "
          f"mem={words_from_memory(r2.memory)}, r7={r2.regs[7]}, "
          f"r8={r2.regs[8]}, r9={r2.regs[9]}")
    print(f"  顺序解释器: halted={seq2.halted}, fault={seq2.fault}, "
          f"mem={words_from_memory(seq2.memory)}, r9={seq2.regs[9]}")
    print(f"  提交序列: 流水线 {[t.pc for t in r2.trace]} / "
          f"顺序 {[t.pc for t in seq2.trace]}")
    assert r2.halted and r2.fault is None
    assert words_from_memory(r2.memory) == [1234, 0]
    assert (r2.regs[7], r2.regs[8], r2.regs[9]) == (0, 0, 100)
    assert [t.pc for t in r2.trace] == [t.pc for t in seq2.trace]
    print("  错误路径 store 未写、HALT 未停机、非法访存未报异常：符合预期")

    print()
    print("全部对照通过：预测只改变时序/统计，架构结果与无预测顺序机逐条一致。")


if __name__ == "__main__":
    main()
