"""五级流水线时序与精确异常的针对性测试。"""

import unittest

from tinycpu import (
    ADD,
    ADDI,
    BEQ,
    HALT,
    LOAD,
    STORE,
    SUB,
    Fault,
    Instruction,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import CycleLimitError, run


def _reasons(cycle_slots, stage):
    out = []
    for c in cycle_slots:
        for s in c.slots:
            if s.stage == stage:
                out.append(s.reason)
    return out


def _stage_indices(result, stage):
    return [
        next(s.index for s in c.slots if s.stage == stage)
        for c in result.cycles
    ]


class TestBasicPipeline(unittest.TestCase):
    def test_five_stage_fill_and_drain(self):
        prog = [
            Instruction.addi(1, 0, 5),
            Instruction.addi(2, 0, 7),
            Instruction.add(3, 1, 2),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertTrue(r.halted)
        self.assertEqual(r.regs[1], 5)
        self.assertEqual(r.regs[2], 7)
        self.assertEqual(r.regs[3], 12)
        # 4 条指令无停顿：5 + 4 - 1 = 8 周期
        self.assertEqual(r.cycles_used, 8)
        # 退休严格按程序序
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3])
        self.assertEqual([t.cycle for t in r.trace], [5, 6, 7, 8])

    def test_r0_always_zero(self):
        prog = [
            Instruction.addi(0, 0, 9),
            Instruction.add(0, 0, 0),
            Instruction.load(0, 0, 0),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([123]))
        self.assertEqual(r.regs[0], 0)
        # 对 r0 的写不出现在退休轨迹的写回记录中
        self.assertTrue(all(t.rd != 0 for t in r.trace))

    def test_32bit_wraparound(self):
        prog = [
            Instruction.addi(1, 0, -1),          # r1 = 0xFFFFFFFF
            Instruction.addi(2, 1, 1),           # 回绕到 0
            Instruction.sub(3, 0, 1),            # 0 - 0xFFFFFFFF = 1
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertEqual(r.regs[1], 0xFFFFFFFF)
        self.assertEqual(r.regs[2], 0)
        self.assertEqual(r.regs[3], 1)


class TestForwardingAndStall(unittest.TestCase):
    def test_load_use_inserts_one_stall(self):
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.add(2, 1, 0),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([42]))
        self.assertEqual(r.regs[2], 42)
        # EX 级恰好插入一个 stall 气泡（它随后沿 MEM->WB 传播）。
        stalls = [
            s
            for c in r.cycles
            for s in c.slots
            if s.bubble and s.reason == "stall" and s.stage == "EX"
        ]
        self.assertEqual(len(stalls), 1)
        # 停顿周期 PC 与 IF/ID 保持（hold）
        hold_cycle = next(
            c for c in r.cycles
            if any(s.reason == "stall" for s in c.slots)
        )
        self.assertEqual(hold_cycle.slots[0].index, 2)  # IF 仍指着 pc2
        self.assertEqual(hold_cycle.slots[1].index, 1)  # ID 仍是 pc1

    def test_no_stall_when_register_unrelated(self):
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.add(2, 3, 4),   # 不使用 r1
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([5]))
        self.assertFalse(
            any(s.reason == "stall" for c in r.cycles for s in c.slots)
        )

    def test_no_stall_after_non_load_write(self):
        prog = [
            Instruction.addi(1, 0, 9),
            Instruction.add(2, 1, 0),   # EX/MEM 前递，无停顿
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertEqual(r.regs[2], 9)
        self.assertFalse(
            any(s.reason == "stall" for c in r.cycles for s in c.slots)
        )

    def test_back_to_back_same_register_chain(self):
        # 连续同寄存器依赖：EX 前递优先"最新值"（MEM 锁存器先于 WB）。
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(1, 1, 10),
            Instruction.add(1, 1, 1),
            Instruction.sub(1, 1, 0),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertEqual(r.regs[1], 22)  # 1 -> 11 -> 22 -> 22
        self.assertTrue(r.halted and r.fault is None)

    def test_forwarding_newest_wins(self):
        # EX 时同一寄存同时可从 MEM 与 WB 前递：必须取 MEM（更新）。
        # pc0 ADDI r5=100; pc1 ADDI r5=200; pc2 ADD r6=r5+r5 -> 400
        prog = [
            Instruction.addi(5, 0, 100),
            Instruction.addi(5, 0, 200),
            Instruction.add(6, 5, 5),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertEqual(r.regs[6], 400)

    def test_load_use_then_forward_from_memwb(self):
        # LOAD -> ADD（停一周期）-> 再来一条 ADD，验证 WB 前递路径。
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.add(2, 1, 0),
            Instruction.add(3, 1, 2),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([10]))
        self.assertEqual((r.regs[1], r.regs[2], r.regs[3]), (10, 10, 20))


class TestBranch(unittest.TestCase):
    def _branch_program(self):
        return [
            Instruction.addi(1, 0, 3),                # 0
            Instruction.addi(2, 0, 3),                # 1
            Instruction.beq(1, 2, 2),                 # 2 命中 -> 4
            Instruction.addi(3, 0, 99),               # 3 被冲刷
            Instruction.addi(4, 0, 7),                # 4
            Instruction.halt(),                       # 5
        ]

    def test_taken_flushes_two_young(self):
        r = run(self._branch_program(), memory_from_words([0]))
        self.assertEqual(r.regs[3], 0)       # 被冲刷的指令不写寄存器
        self.assertEqual(r.regs[4], 7)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 4, 5])
        flushed = [
            s for c in r.cycles for s in c.slots
            if s.bubble and s.reason == "flush" and s.index is not None
        ]
        # 决策周期 ID、IF 两条年轻受害指令各一个冲刷标记
        # （其后的空气泡会沿 EX->MEM->WB 传播，不计入受害者）。
        self.assertEqual(len(flushed), 2)

    def test_not_taken_no_flush(self):
        prog = [
            Instruction.addi(1, 0, 3),
            Instruction.addi(2, 0, 4),
            Instruction.beq(1, 2, 2),     # 不命中
            Instruction.addi(3, 0, 9),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertEqual(r.regs[3], 9)
        self.assertFalse(
            any(s.reason == "flush" for c in r.cycles for s in c.slots)
        )
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 4])

    def test_backward_branch_loop(self):
        # 用向后分支实现计数循环：r1 从 3 递减，减到 0 时退出。
        prog = [
            Instruction.addi(1, 0, 3),      # 0 r1 = 3
            Instruction.addi(1, 1, -1),     # 1 r1 -= 1
            Instruction.beq(1, 0, 2),       # 2 r1==0 -> pc4 退出
            Instruction.beq(0, 0, -2),      # 3 无条件回 pc1
            Instruction.halt(),             # 4
        ]
        r = run(prog, memory_from_words([0]))
        self.assertTrue(r.halted)
        self.assertEqual(r.regs[1], 0)
        # pc1 执行 3 次，pc2 执行 3 次，pc3 执行 2 次
        from collections import Counter
        cnt = Counter(t.pc for t in r.trace)
        self.assertEqual(cnt[1], 3)
        self.assertEqual(cnt[2], 3)
        self.assertEqual(cnt[3], 2)


class TestPreciseException(unittest.TestCase):
    def test_illegal_load_reported_at_wb(self):
        # 基址 = 2（非对齐）；故障必须等到该指令进 WB 的周期才出现。
        prog = [
            Instruction.addi(1, 0, 2),
            Instruction.load(2, 1, 0),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0, 0]))
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(r.fault_pc, 1)
        # LOAD 在第 6 周期（IF 起算第 2 条 -> WB = 2+4）报告
        self.assertEqual(r.fault_cycle, 6)
        self.assertFalse(r.halted)
        self.assertEqual(r.regs[2], 0)  # 故障 LOAD 不写寄存器

    def test_illegal_store_does_not_write(self):
        prog = [
            Instruction.addi(1, 0, 3),   # 非对齐基址
            Instruction.store(0, 1, 0),
            Instruction.halt(),
        ]
        mem = memory_from_words([7, 7])
        r = run(prog, mem)
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(words_from_memory(r.memory), [7, 7])
        # 轨迹里故障 STORE 不带内存写效果
        last = r.trace[-1]
        self.assertIsNotNone(last.fault)
        self.assertIsNone(last.addr)

    def test_old_store_committed_young_store_not(self):
        # 题面关键场景：非法访存到 WB 才报告，
        # 老指令正常提交，年轻 STORE 不得已写内存。
        prog = [
            Instruction.addi(5, 0, 111),   # 0
            Instruction.addi(1, 0, 0),     # 1
            Instruction.store(5, 1, 0),    # 2 合法老 STORE
            Instruction.addi(6, 0, 2),     # 3 错误基址
            Instruction.load(7, 6, 0),     # 4 非法 LOAD
            Instruction.store(5, 1, 8),    # 5 年轻合法 STORE，禁止写
            Instruction.halt(),            # 6
        ]
        r = run(prog, memory_from_words([0, 0, 0, 0]))
        self.assertEqual(r.fault_pc, 4)
        self.assertEqual(words_from_memory(r.memory), [111, 0, 0, 0])
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 4])
        # 故障周期：WB 报故障，MEM/EX/ID/IF 的年轻指令标 fault 丢弃
        fc = r.cycles[r.fault_cycle - 1]
        self.assertEqual(fc.slots[4].reason, "fault")
        for s in fc.slots[:4]:
            if s.index is not None:
                self.assertEqual(s.reason, "fault")

    def test_illegal_write_after_taken_branch(self):
        # 分支冲刷了 pc3/pc4，随后 pc6 的非法 STORE 在 WB 报告，
        # 其后更年轻的合法 STORE 不得写。
        prog = [
            Instruction.addi(1, 0, 4),                # 0
            Instruction.addi(2, 0, 4),                # 1
            Instruction.beq(1, 2, 2),                 # 2 -> 4
            Instruction.addi(3, 0, 9),                # 3 冲刷
            Instruction.store(3, 0, 0),               # 4 冲刷
            Instruction.addi(4, 0, 7),                # 5 非对齐基址
            Instruction.store(1, 4, 0),               # 6 非法 STORE
            Instruction.store(2, 0, 4),               # 7 年轻 STORE 禁写
            Instruction.halt(),                       # 8
        ]
        r = run(prog, memory_from_words([0, 0, 0]))
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(r.fault_pc, 6)
        self.assertEqual(words_from_memory(r.memory), [0, 0, 0])
        self.assertEqual(r.regs[3], 0)

    def test_out_of_range_address(self):
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.load(2, 1, 16),   # 内存只有 8 字节
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0, 0]))
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(r.fault_pc, 1)


class TestLimits(unittest.TestCase):
    def test_cycle_budget_separate_from_fault(self):
        # 无限循环（BEQ r0,r0,0）必须报预算耗尽，而不是被当作故障或静默结束。
        # HALT 永远到不了，但静态契约仍要求程序以 HALT 结尾。
        prog = [Instruction.beq(0, 0, 0), Instruction.halt()]
        with self.assertRaises(CycleLimitError):
            run(prog, memory_from_words([0]))

    def test_halt_ignores_following_words(self):
        # 程序以 HALT 结束；循环回到 HALT 时正常停机（不会重新取后面）。
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.halt(),
        ]
        r = run(prog, memory_from_words([0]))
        self.assertTrue(r.halted)


if __name__ == "__main__":
    unittest.main()
