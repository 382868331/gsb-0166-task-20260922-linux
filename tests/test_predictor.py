"""两位饱和计数器分支预测器的测试。

覆盖：IF 查表选 PC、EX 判错冲刷年轻指令、WB 提交才训练、
被冲掉的分支不污染计数器、错误路径的 store/HALT/非法访存不变成架构结果、
预测/错判/冲刷统计、PC 直接索引别名碰撞，以及与无预测顺序解释器的
逐条提交记录对照。
"""

import random
import unittest

from tinycpu import (
    Instruction,
    memory_from_words,
)
from tinycpu.pipeline import BranchPredictor, run
from tinycpu.sequential import run_sequential


def _loop_program():
    # r1 从 3 递减到 0：pc2 三次（N,N,T），pc3 无条件回跳两次（T,T）。
    return [
        Instruction.addi(1, 0, 3),      # 0
        Instruction.addi(1, 1, -1),     # 1
        Instruction.beq(1, 0, 2),       # 2 减到 0 -> 4
        Instruction.beq(0, 0, -2),      # 3 无条件回 1
        Instruction.halt(),             # 4
    ]


class TestPredictorConstruction(unittest.TestCase):
    def test_rejects_bad_entries(self):
        with self.assertRaises(ValueError):
            BranchPredictor(0)
        with self.assertRaises(ValueError):
            BranchPredictor(-4)
        with self.assertRaises(TypeError):
            BranchPredictor(True)

    def test_rejects_bad_initial(self):
        with self.assertRaises(ValueError):
            BranchPredictor(4, -1)
        with self.assertRaises(ValueError):
            BranchPredictor(4, 4)
        with self.assertRaises(TypeError):
            BranchPredictor(4, False)

    def test_run_rejects_bad_predictor_type(self):
        with self.assertRaises(TypeError):
            run([Instruction.halt()], memory_from_words([0]), None, predictor=3)

    def test_direct_index_alias(self):
        bp = BranchPredictor(4, 0)
        self.assertEqual(bp.index_of(0), 0)
        self.assertEqual(bp.index_of(4), 0)   # 别名碰撞
        self.assertEqual(bp.index_of(7), 3)
        self.assertEqual(bp.counters, [0, 0, 0, 0])


class TestPredictionStats(unittest.TestCase):
    def test_no_predictor_stats_are_none(self):
        r = run(_loop_program(), memory_from_words([0]))
        self.assertIsNone(r.predictions)
        self.assertIsNone(r.mispredictions)
        self.assertIsNone(r.flushed_instructions)

    def test_loop_weak_not_taken_seed(self):
        # 初值 1（弱不跳）：pc2 序列 N,N,T，第一次 N 即预测正确，
        # 随后训练到 taken（1->0->1->3），只有最后一次判错；
        # pc3 总跳，首访弱不跳、被 pc2 别名无关地训练到 3，判错 1 次。
        bp = BranchPredictor(8, 1)
        r = run(_loop_program(), memory_from_words([0]), None, bp)
        self.assertEqual(r.predictions, 5)
        self.assertEqual(r.mispredictions, 2)
        # 一次冲刷发生在 IF 排空边界（受害槽只有 ID 一条），共 3 条。
        self.assertEqual(r.flushed_instructions, 3)
        # 提交后：pc2 = 1（N:1->0, N:0->0, T:0->1），pc3 = 3（1->2->3）。
        self.assertEqual(bp.counters[2], 1)
        self.assertEqual(bp.counters[3], 3)

    def test_loop_taken_seed_warms_up(self):
        bp = BranchPredictor(8, 3)
        r = run(_loop_program(), memory_from_words([0]), None, bp)
        # pc2 预测跳：N、N 错，T 时计数已降到 1（预测不跳）仍错，
        # 三次全错（3->2->1->2）；pc3 全对（恒为 3）。4 条冲刷。
        self.assertEqual((r.predictions, r.mispredictions), (5, 3))
        self.assertEqual(r.flushed_instructions, 4)
        self.assertEqual(bp.counters[2], 2)
        self.assertEqual(bp.counters[3], 3)

    def test_flushed_count_matches_slot_victims(self):
        bp = BranchPredictor(8, 1)
        r = run(_loop_program(), memory_from_words([0]), None, bp)
        victims = [
            s for c in r.cycles for s in c.slots
            if s.bubble and s.reason == "mispredict" and s.index is not None
        ]
        self.assertEqual(len(victims), r.flushed_instructions)
        for s in victims:
            self.assertIn(s.stage, ("IF", "ID"))


class TestNoWrongPathSideEffects(unittest.TestCase):
    def test_wrong_path_store_does_not_write(self):
        # pc2 实际跳转到 6，但预测为不跳 -> 错误路径 pc3 的 STORE 被冲，
        # 错误路径 pc4/pc5 也不得执行。
        prog = [
            Instruction.addi(1, 0, 5),          # 0
            Instruction.addi(2, 0, 5),          # 1
            Instruction.beq(1, 2, 4),           # 2 -> 6
            Instruction.store(1, 0, 0),         # 3 错误路径 store，禁写
            Instruction.addi(3, 0, 99),         # 4 错误路径
            Instruction.addi(4, 0, 77),         # 5 错误路径
            Instruction.halt(),                 # 6
        ]
        mem = memory_from_words([1234])
        bp = BranchPredictor(8, 0)
        r = run(prog, bytearray(mem), None, bp)
        self.assertTrue(r.halted)
        self.assertIsNone(r.fault)
        self.assertEqual(r.memory, bytes(mem))
        self.assertEqual(r.regs[3], 0)
        self.assertEqual(r.regs[4], 0)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 6])
        self.assertEqual(r.mispredictions, 1)
        self.assertEqual(r.flushed_instructions, 2)

    def test_wrong_path_halt_does_not_halt_machine(self):
        # 分支实际跳转，预测不跳；错误路径 pc3 是 HALT，绝不能提前停机。
        prog = [
            Instruction.addi(1, 0, 5),          # 0
            Instruction.addi(2, 0, 5),          # 1
            Instruction.beq(1, 2, 3),           # 2 -> 5
            Instruction.halt(),                 # 3 错误路径 HALT
            Instruction.addi(9, 0, 42),         # 4
            Instruction.addi(8, 0, 7),          # 5 正确路径
            Instruction.halt(),                 # 6
        ]
        bp = BranchPredictor(8, 0)
        r = run(prog, memory_from_words([0]), None, bp)
        self.assertTrue(r.halted)
        self.assertIsNone(r.fault_pc)
        self.assertEqual(r.regs[9], 0)   # pc4 被冲
        self.assertEqual(r.regs[8], 7)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 5, 6])

    def test_wrong_path_bad_access_is_flushed_no_fault(self):
        # 错误路径 pc4 是非法 STORE（基址 3 非对齐），它在 EX 前被冲掉，
        # 既不能写内存，也不能报告 ADDR 异常；正确路径继续到 HALT。
        prog = [
            Instruction.addi(1, 0, 5),          # 0
            Instruction.addi(2, 0, 5),          # 1
            Instruction.beq(1, 2, 4),           # 2 -> 6
            Instruction.addi(6, 0, 3),          # 3 错误路径：非对齐基址
            Instruction.store(1, 6, 0),         # 4 错误路径非法 STORE
            Instruction.addi(7, 0, 9),          # 5 错误路径
            Instruction.halt(),                 # 6
        ]
        mem = memory_from_words([9, 9])
        bp = BranchPredictor(8, 0)
        r = run(prog, bytearray(mem), None, bp)
        self.assertIsNone(r.fault)
        self.assertTrue(r.halted)
        self.assertEqual(r.memory, bytes(mem))
        self.assertEqual(r.regs[7], 0)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 6])

    def test_flushed_branch_does_not_train_counter(self):
        # pc2 预测不跳且实际不跳（预测正确）；pc3 必跳但被预测为不跳，
        # 错误路径 pc4（一条 BEQ）被冲刷——它绝不允许训练自己的表项。
        prog = [
            Instruction.addi(1, 0, 1),          # 0
            Instruction.addi(2, 0, 9),          # 1
            Instruction.beq(1, 2, 2),           # 2 不跳 -> 3
            Instruction.beq(0, 0, 2),           # 3 必跳 -> 5
            Instruction.beq(0, 0, 0),           # 4 被冲刷，绝不训练
            Instruction.halt(),                 # 5
        ]
        bp = BranchPredictor(8, 0)
        r = run(prog, memory_from_words([0]), None, bp)
        self.assertTrue(r.halted)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 5])
        self.assertEqual(bp.counters[4], 0)     # 被冲分支保持初值
        self.assertEqual(bp.counters[3], 1)     # pc3 提交后 0->1
        self.assertEqual(r.flushed_instructions, 2)


class TestPCAlias(unittest.TestCase):
    def test_aliased_entries_change_misprediction_rate(self):
        # 循环中 pc2 与 pc3 在 1 项表中互相别名，在 8 项表中独立。
        prog = _loop_program()
        r_alias = run(prog, bytearray(memory_from_words([0])), None,
                      BranchPredictor(1, 1))
        r_split = run(prog, bytearray(memory_from_words([0])), None,
                      BranchPredictor(8, 1))
        # 两者架构结果完全一致，但别名表的错判率与独立表不同。
        self.assertEqual(r_alias.regs, r_split.regs)
        self.assertNotEqual(r_alias.mispredictions, r_split.mispredictions)
        self.assertEqual(r_alias.predictions, 5)
        self.assertEqual(r_split.predictions, 5)


class TestPredictorEquivalence(unittest.TestCase):
    """接入预测器后与无预测顺序解释器逐条核对提交记录。"""

    def _assert_matches_sequential(self, prog, mem, init=None):
        seq = run_sequential(prog, bytearray(mem), init)
        for entries in (1, 2, 3, 8, 16):
            for initial in range(4):
                bp = BranchPredictor(entries, initial)
                got = run(prog, bytearray(mem), init, bp)
                self.assertEqual(got.halted, seq.halted, (entries, initial))
                self.assertEqual(got.fault, seq.fault, (entries, initial))
                self.assertEqual(got.fault_pc, seq.fault_pc, (entries, initial))
                self.assertEqual(got.regs, seq.regs, (entries, initial))
                self.assertEqual(got.memory, seq.memory, (entries, initial))
                self.assertEqual(len(got.trace), len(seq.trace), (entries, initial))
                for tp, ts in zip(got.trace, seq.trace):
                    self.assertEqual(
                        (tp.pc, tp.op, tp.fault, tp.rd, tp.value,
                         tp.addr, tp.mem_value),
                        (ts.pc, ts.op, ts.fault, ts.rd, ts.value,
                         ts.addr, ts.mem_value),
                        (entries, initial, tp, ts),
                    )
                self.assertGreaterEqual(got.mispredictions, 0)
                self.assertLessEqual(
                    got.flushed_instructions, 2 * got.predictions
                )

    def test_handwritten_loop_matches(self):
        self._assert_matches_sequential(_loop_program(), memory_from_words([0]))

    def test_handwritten_wrong_path_store_matches(self):
        prog = [
            Instruction.addi(1, 0, 5),
            Instruction.addi(2, 0, 5),
            Instruction.beq(1, 2, 4),
            Instruction.store(1, 0, 0),
            Instruction.addi(3, 0, 99),
            Instruction.addi(4, 0, 77),
            Instruction.halt(),
        ]
        self._assert_matches_sequential(prog, memory_from_words([1234]))

    def test_random_terminating_loops(self):
        # 确定性随机：结构化计数循环（含后向分支），50 个种子，
        # 多种表容量/初值，全部与顺序解释器逐条一致。
        rng = random.Random(20260926)
        for seed in range(50):
            iters = 1 + seed % 6
            prog = [Instruction.addi(31, 0, iters)]
            body = 1 + rng.randrange(4)
            for _ in range(body):
                op = rng.choice(["addi", "add", "sub"])
                rd = rng.randrange(1, 30)
                if op == "addi":
                    prog.append(Instruction.addi(rd, rng.randrange(30),
                                                 rng.randrange(-8, 9)))
                else:
                    cls = Instruction.add if op == "add" else Instruction.sub
                    prog.append(cls(rd, rng.randrange(30), rng.randrange(30)))
            k1 = len(prog)
            prog.append(Instruction.addi(31, 31, -1))
            prog.append(Instruction.beq(31, 0, 2))
            prog.append(Instruction.beq(0, 0, 1 - (k1 + 2)))
            prog.append(Instruction.halt())
            words = [(seed * 131 + i * 17) & 0xFFFFFFFF for i in range(8)]
            self._assert_matches_sequential(
                prog, memory_from_words(words), [0] * 32
            )


if __name__ == "__main__":
    unittest.main()
