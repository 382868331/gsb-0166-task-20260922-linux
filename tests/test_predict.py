"""分支预测增量测试：两位饱和计数器预测表接入五级流水线。

覆盖：
* BranchPredictor 自身的校验与饱和/别名语义；
* 循环程序：预测统计（预测次数/错判/冲刷）与逐条退休记录对账顺序参考；
* 两个 PC 别名碰撞同一表项；
* 错误路径上的 STORE / HALT / 非法访存不得成为架构结果；
* 训练只由提交的分支触发（被故障丢弃的分支不污染计数器）；
* 未启用预测时基线行为不变（prediction 为 None）；
* 确定性随机程序 + 循环程序的差分对账（复用 test_equivalence 的生成器）。
"""

import unittest

from tinycpu import (
    BranchPredictor,
    Fault,
    Instruction,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential
from test_equivalence import (
    MEM_BYTES,
    effects_equal,
    make_loop_program,
    make_program,
)


def sum_loop_program():
    """1+2+3+4+5 的计数循环：两个 BEQ（条件退出 + 无条件回跳）。"""
    return [
        Instruction.addi(2, 0, 5),    # 0: r2 = 5（上限）
        Instruction.addi(1, 0, 0),    # 1: r1 = i
        Instruction.addi(3, 0, 0),    # 2: r3 = sum
        Instruction.addi(1, 1, 1),    # 3: i += 1
        Instruction.add(3, 3, 1),     # 4: sum += i
        Instruction.beq(1, 2, 2),     # 5: i == 5 时退出到 7
        Instruction.beq(0, 0, -3),    # 6: 否则回 3
        Instruction.halt(),           # 7
    ]


def assert_matches_sequential(testcase, result, seq):
    """逐条核对提交记录（退休轨迹）、最终寄存器与内存。"""
    testcase.assertEqual(len(result.trace), len(seq.trace))
    for t_pipe, t_seq in zip(result.trace, seq.trace):
        testcase.assertTrue(
            effects_equal(t_pipe, t_seq),
            f"退休记录不一致: 流水线={t_pipe} 顺序={t_seq}",
        )
    testcase.assertEqual(result.regs, seq.regs)
    testcase.assertEqual(result.memory, seq.memory)
    testcase.assertEqual(result.halted, seq.halted)
    testcase.assertEqual(result.fault, seq.fault)
    testcase.assertEqual(result.fault_pc, seq.fault_pc)


class TestPredictorUnit(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            BranchPredictor(0)
        with self.assertRaises(TypeError):
            BranchPredictor(True)          # bool 冒充 int
        with self.assertRaises(TypeError):
            BranchPredictor(2.5)
        with self.assertRaises(ValueError):
            BranchPredictor(4, 4)          # 初值超出 2 位
        with self.assertRaises(ValueError):
            BranchPredictor(4, -1)
        with self.assertRaises(ValueError):
            BranchPredictor(4, float("nan"))
        with self.assertRaises(TypeError):
            BranchPredictor(4, True)

    def test_saturating_counter(self):
        p = BranchPredictor(4, initial=0)
        self.assertFalse(p.predict(0))
        for expect in (1, 2, 3, 3, 3):     # 饱和在 3
            p.update(0, True)
            self.assertEqual(p.counter(0), expect)
        self.assertTrue(p.predict(0))
        for expect in (2, 1, 0, 0, 0):     # 饱和在 0
            p.update(0, False)
            self.assertEqual(p.counter(0), expect)
        self.assertFalse(p.predict(0))

    def test_direct_index_aliasing(self):
        p = BranchPredictor(2, initial=0)
        p.update(0, True)                  # pc0 与 pc2 共用表项 0
        self.assertEqual(p.counter(2), 1)  # 别名：同一计数器
        self.assertEqual(p.counter(1), 0)  # 表项 1 不受影响
        with self.assertRaises(TypeError):
            p.update(0, 1)                 # taken 必须是 bool
        with self.assertRaises(ValueError):
            p.predict(-1)

    def test_counters_snapshot_is_copy(self):
        p = BranchPredictor(2, initial=1)
        snap = p.counters
        p.update(0, True)
        self.assertEqual(snap, (1, 1))
        self.assertEqual(p.counters, (2, 1))


class TestLoopPrediction(unittest.TestCase):
    def test_loop_stats_and_equivalence(self):
        prog = sum_loop_program()
        mem = memory_from_words([0] * 4)
        pred = BranchPredictor(8, initial=1)
        result = run(prog, mem, predictor=pred)
        seq = run_sequential(prog, memory_from_words([0] * 4))
        assert_matches_sequential(self, result, seq)
        self.assertEqual(result.regs[3], 15)          # 1+2+3+4+5
        stats = result.prediction
        self.assertIsNotNone(stats)
        # pc5 条件分支取指 5 次；pc6 回跳取指 4 次；
        # pc5 最后一次错判为不命中时，错误路径多取了一次 pc6。
        self.assertEqual(stats.predictions, 10)
        # pc5 末次退出错判 1 次；pc6 首次（计数器初值 1 预测不命中）错判 1 次。
        self.assertEqual(stats.mispredictions, 2)
        # pc5 末次错判冲刷 ID、IF 两条；pc6 首次错判时 IF 已越过程序末尾，
        # 只有 ID 一名受害者，合计 3 条。
        self.assertEqual(stats.flushed, 3)
        # 训练后：pc5 计数器被末次命中抬到 1；pc6 学到强命中 3。
        self.assertEqual(pred.counter(5), 1)
        self.assertEqual(pred.counter(6), 3)

    def test_no_predictor_keeps_baseline(self):
        prog = sum_loop_program()
        result = run(prog, memory_from_words([0] * 4))
        self.assertIsNone(result.prediction)
        self.assertEqual(result.regs[3], 15)


class TestAliasing(unittest.TestCase):
    def test_two_pcs_share_entry(self):
        """表容量 1：两个 BEQ 必然别名，结果仍须与顺序参考逐条一致。"""
        prog = sum_loop_program()
        mem = memory_from_words([0] * 4)
        pred = BranchPredictor(1, initial=1)
        result = run(prog, mem, predictor=pred)
        seq = run_sequential(prog, memory_from_words([0] * 4))
        assert_matches_sequential(self, result, seq)
        stats = result.prediction
        # 别名互相干扰，错判多于 8 表项时的 2 次。
        self.assertGreater(stats.mispredictions, 2)
        self.assertGreaterEqual(stats.flushed, stats.mispredictions)


class TestWrongPathEffects(unittest.TestCase):
    def test_wrong_path_store_and_fault_not_committed(self):
        """预测命中跳过错判路径：错误路径上的非法 STORE 不得报错/写内存。"""
        prog = [
            Instruction.addi(1, 0, 1),      # 0: r1 = 1
            Instruction.addi(2, 0, 2),      # 1: r2 = 2
            Instruction.beq(1, 2, 3),       # 2: 实际不命中；预测命中到 5
            Instruction.store(1, 0, 0),     # 3: mem[0] = 1（正确路径）
            Instruction.beq(0, 0, 2),       # 4: 正确路径跳过错误路径到 6
            Instruction.store(2, 0, 252),   # 5: 仅错误路径，非法地址
            Instruction.halt(),             # 6
        ]
        mem = memory_from_words([0] * 4)
        pred = BranchPredictor(4, initial=3)   # 强命中：取 pc4 的错误路径
        result = run(prog, mem, predictor=pred)
        seq = run_sequential(prog, memory_from_words([0] * 4))
        assert_matches_sequential(self, result, seq)
        self.assertTrue(result.halted)
        self.assertIsNone(result.fault)        # 错误路径的非法 STORE 未报告
        self.assertEqual(words_from_memory(result.memory), [1, 0, 0, 0])
        stats = result.prediction
        self.assertEqual(stats.predictions, 2)      # pc2 与 pc4 各预测一次
        self.assertEqual(stats.mispredictions, 1)   # 仅 pc2 错判
        self.assertEqual(stats.flushed, 2)

    def test_wrong_path_halt_not_committed(self):
        """错误路径取到 HALT：分支解析后冲刷，机器继续走正确路径。"""
        prog = [
            Instruction.addi(1, 0, 1),      # 0
            Instruction.addi(2, 0, 2),      # 1
            Instruction.beq(1, 2, 2),       # 2: 实际不命中；预测命中到 4
            Instruction.addi(3, 0, 7),      # 3: 正确路径
            Instruction.halt(),             # 4
        ]
        mem = memory_from_words([0] * 4)
        pred = BranchPredictor(4, initial=3)
        result = run(prog, mem, predictor=pred)
        seq = run_sequential(prog, memory_from_words([0] * 4))
        assert_matches_sequential(self, result, seq)
        self.assertTrue(result.halted)
        self.assertEqual(result.regs[3], 7)    # HALT 没有提前提交

    def test_flushed_branch_does_not_train(self):
        """老指令故障丢弃年轻分支：已解析未提交的分支不训练计数器。"""
        prog = [
            Instruction.load(1, 0, 252),    # 0: 非法 LOAD，WB（周期 5）报告
            Instruction.beq(0, 0, 2),       # 1: 命中到 3；周期 4 在 EX 解析
            Instruction.addi(5, 0, 9),      # 2: 被冲刷
            Instruction.halt(),             # 3
        ]
        mem = memory_from_words([0] * 4)
        pred = BranchPredictor(4, initial=1)
        result = run(prog, mem, predictor=pred)
        seq = run_sequential(prog, memory_from_words([0] * 4))
        assert_matches_sequential(self, result, seq)
        self.assertEqual(result.fault, Fault.ADDR)
        self.assertEqual(result.fault_pc, 0)
        stats = result.prediction
        self.assertEqual(stats.predictions, 1)
        self.assertEqual(stats.mispredictions, 1)   # EX 已发现方向错
        self.assertEqual(stats.flushed, 2)
        # 关键：BEQ 在 MEM 级被故障丢弃，从未提交，计数器保持初值。
        self.assertEqual(pred.counter(1), 1)
        self.assertEqual(words_from_memory(result.memory), [0, 0, 0, 0])


class TestDifferential(unittest.TestCase):
    """带预测与顺序参考的差分对账（复用 test_equivalence 的生成器）。"""

    def test_random_programs_with_predictor(self):
        for seed in range(200):
            length = 6 + seed % 12
            prog, init = make_program(seed, length)
            mem = memory_from_words([0] * (MEM_BYTES // 4))
            # 小表强制别名碰撞；初值随种子变化覆盖四种起点。
            pred = BranchPredictor(1 + seed % 4, initial=seed % 4)
            result = run(prog, mem, init, predictor=pred)
            seq = run_sequential(prog, memory_from_words([0] * (MEM_BYTES // 4)), init)
            with self.subTest(seed=seed):
                assert_matches_sequential(self, result, seq)
                stats = result.prediction
                self.assertIsNotNone(stats)
                self.assertGreaterEqual(stats.predictions, stats.mispredictions)
                self.assertGreaterEqual(stats.flushed, stats.mispredictions)

    def test_loop_programs_with_predictor(self):
        for seed in range(60):
            prog, init = make_loop_program(seed)
            mem = memory_from_words([0] * (MEM_BYTES // 4))
            pred = BranchPredictor(1 + seed % 8, initial=seed % 4)
            result = run(prog, mem, init, predictor=pred)
            seq = run_sequential(prog, memory_from_words([0] * (MEM_BYTES // 4)), init)
            with self.subTest(seed=seed):
                assert_matches_sequential(self, result, seq)

    def test_predictor_type_checked(self):
        with self.assertRaises(TypeError):
            run(sum_loop_program(), memory_from_words([0] * 4), predictor=object())


if __name__ == "__main__":
    unittest.main()

