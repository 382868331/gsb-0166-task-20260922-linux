"""显式中断请求与 IRET 的测试。

覆盖：退休边界接纳、保存 PC、年轻指令冲刷（含被冲刷 STORE 不写内存）、
屏蔽与 pending 合并、IRET 返回后解除屏蔽、异常优先、HALT 后不再接纳、
非法向量拒绝注入；并对 load-use 停顿、分支冲刷、异常与中断同拍三个
小程序，用独立顺序参考在相同退休边界注入事件做差分比较。
"""

import random
import unittest

from tinycpu import (
    Fault,
    Instruction,
    InterruptController,
    ProgramError,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential


def build(main, handler):
    """程序布局：pc0 用 BEQ 跳过处理器；handler 从 pc1 开始；main 紧随其后。

    返回 (prog, vector)。main 必须以 HALT 结尾（静态契约）。
    """
    jump = Instruction.beq(0, 0, len(handler) + 1)
    return [jump, *handler, *main], 1


def run_both(prog, mem, pulse_boundaries, vector):
    """同一程序、同一边界注入计划，分别跑流水线与顺序参考。

    控制器被运行消费（pending 位变化），故两侧各用一个新控制器。
    """
    pipe = run(
        prog,
        bytearray(mem),
        interrupts=InterruptController(pulse_boundaries),
        interrupt_vector=vector,
    )
    seq = run_sequential(
        prog,
        bytearray(mem),
        interrupts=InterruptController(pulse_boundaries),
        interrupt_vector=vector,
    )
    return pipe, seq


def assert_same_arch_state(tc, pipe, seq):
    """寄存器、内存、退休序列、中断事件（边界/保存PC/向量）完全一致。"""
    tc.assertEqual(pipe.regs, seq.regs)
    tc.assertEqual(pipe.memory, seq.memory)
    tc.assertEqual([t.pc for t in pipe.trace], [t.pc for t in seq.trace])
    tc.assertEqual(
        [(t.rd, t.value, t.addr, t.mem_value, t.fault) for t in pipe.trace],
        [(t.rd, t.value, t.addr, t.mem_value, t.fault) for t in seq.trace],
    )
    tc.assertEqual(
        [(e.boundary, e.saved_pc, e.vector_pc) for e in pipe.irq_events],
        [(e.boundary, e.saved_pc, e.vector_pc) for e in seq.irq_events],
    )
    tc.assertEqual(pipe.irq_pending, seq.irq_pending)
    tc.assertEqual(pipe.irq_masked, seq.irq_masked)


class TestAcceptance(unittest.TestCase):
    def test_basic_accept_save_pc_flush_resume(self):
        handler = [
            Instruction.addi(4, 0, 1),   # pc1 标记进入过处理器
            Instruction.iret(),          # pc2
        ]
        main = [
            Instruction.addi(1, 0, 10),  # pc3
            Instruction.addi(2, 0, 20),  # pc4
            Instruction.add(3, 1, 2),    # pc5
            Instruction.store(3, 0, 0),  # pc6
            Instruction.halt(),          # pc7
        ]
        prog, vector = build(main, handler)
        # 退休顺序：b0=BEQ, b1=pc3, b2=pc4；在 b2 注入 -> 接纳于 pc4 退休之后
        pipe, seq = run_both(prog, memory_from_words([0]), {2}, vector)

        # 保存的 PC = 下一条尚未退休指令 pc5；退休序列可见处理器往返
        self.assertEqual(len(pipe.irq_events), 1)
        ev = pipe.irq_events[0]
        self.assertEqual((ev.boundary, ev.saved_pc, ev.vector_pc), (2, 5, 1))
        self.assertIsNotNone(ev.cycle)
        self.assertEqual([t.pc for t in pipe.trace], [0, 3, 4, 1, 2, 5, 6, 7])
        self.assertEqual(pipe.regs[3], 30)
        self.assertEqual(pipe.regs[4], 1)
        self.assertEqual(words_from_memory(pipe.memory), [30])
        self.assertFalse(pipe.irq_pending)
        self.assertFalse(pipe.irq_masked)
        # 接纳周期：年轻各级被标记 irq 冲刷
        irq_slots = [
            s for c in pipe.cycles for s in c.slots if s.reason == "irq"
        ]
        self.assertTrue(irq_slots)
        assert_same_arch_state(self, pipe, seq)

    def test_prearmed_request_accepted_at_first_boundary(self):
        handler = [Instruction.iret()]
        main = [Instruction.addi(1, 0, 7), Instruction.halt()]
        prog, vector = build(main, handler)
        ctrl = InterruptController()
        ctrl.request()  # 运行前预置：第一个退休边界即可接纳
        pipe = run(prog, memory_from_words([0]), interrupts=ctrl,
                   interrupt_vector=vector)
        self.assertEqual(len(pipe.irq_events), 1)
        self.assertEqual(pipe.irq_events[0].boundary, 0)
        self.assertEqual(pipe.irq_events[0].saved_pc, 2)  # main[0] 的 pc
        self.assertTrue(pipe.halted)
        self.assertEqual(pipe.regs[1], 7)

    def test_pending_merges_to_one_bit(self):
        ctrl = InterruptController(pulse_boundaries={0, 0, 0})
        ctrl.request()
        ctrl.request()
        self.assertTrue(ctrl.pending)  # 只有一位，不计数


class TestLoadUseStallBoundary(unittest.TestCase):
    def test_interrupt_during_load_use(self):
        handler = [
            Instruction.addi(7, 0, 1),   # pc1
            Instruction.iret(),          # pc2
        ]
        main = [
            Instruction.load(1, 0, 0),   # pc3 r1 = mem[0]
            Instruction.add(2, 1, 1),    # pc4 load-use 停一拍
            Instruction.store(2, 0, 4),  # pc5
            Instruction.halt(),          # pc6
        ]
        prog, vector = build(main, handler)
        # b0=BEQ, b1=LOAD(pc3)：LOAD 退休边界接纳，年轻指令（含停顿中的 ADD）被冲刷
        pipe, seq = run_both(prog, memory_from_words([21, 0]), {1}, vector)

        self.assertEqual(len(pipe.irq_events), 1)
        self.assertEqual(pipe.irq_events[0].saved_pc, 4)  # 停顿中的 ADD
        self.assertEqual(pipe.regs[1], 21)
        self.assertEqual(pipe.regs[2], 42)
        self.assertEqual(words_from_memory(pipe.memory), [21, 42])
        # load-use 停顿真实发生过一次（IRET 返回后重执行时数据已在寄存器）
        stalls = [
            s for c in pipe.cycles for s in c.slots
            if s.bubble and s.reason == "stall" and s.stage == "EX"
        ]
        self.assertEqual(len(stalls), 1)
        assert_same_arch_state(self, pipe, seq)


class TestBranchFlushBoundary(unittest.TestCase):
    def test_interrupt_with_taken_branch_in_flight(self):
        handler = [
            Instruction.addi(7, 0, 1),   # pc1
            Instruction.iret(),          # pc2
        ]
        main = [
            Instruction.addi(1, 0, 5),   # pc3
            Instruction.addi(2, 0, 5),   # pc4
            Instruction.beq(1, 2, 2),    # pc5 命中 -> pc7
            Instruction.addi(3, 0, 99),  # pc6 被分支冲刷
            Instruction.store(1, 0, 0),  # pc7
            Instruction.halt(),          # pc8
        ]
        prog, vector = build(main, handler)
        # b0=BEQ(pc0), b1=pc3, b2=pc4：接纳时 BEQ(pc5) 尚在流水线中未决策
        pipe, seq = run_both(prog, memory_from_words([0]), {2}, vector)

        self.assertEqual(len(pipe.irq_events), 1)
        self.assertEqual(pipe.irq_events[0].saved_pc, 5)  # 在飞的 BEQ
        # IRET 返回后 BEQ 重新执行并再次冲刷 pc6
        self.assertEqual(pipe.regs[3], 0)
        self.assertEqual(words_from_memory(pipe.memory), [5])
        self.assertEqual([t.pc for t in pipe.trace], [0, 3, 4, 1, 2, 5, 7, 8])
        # 分支冲刷与中断冲刷都真实出现
        reasons = {s.reason for c in pipe.cycles for s in c.slots}
        self.assertIn("flush", reasons)
        self.assertIn("irq", reasons)
        assert_same_arch_state(self, pipe, seq)


class TestFlushedStoreDoesNotWrite(unittest.TestCase):
    def test_handler_observes_pre_store_memory(self):
        handler = [
            Instruction.load(6, 0, 0),   # pc1 读被冲刷 STORE 的地址
            Instruction.iret(),          # pc2
        ]
        main = [
            Instruction.addi(5, 0, 111),  # pc3
            Instruction.store(5, 0, 0),   # pc4 接纳时在 MEM，被冲刷，不得写
            Instruction.store(6, 0, 4),   # pc5 把处理器读到的值存出来
            Instruction.halt(),           # pc6
        ]
        prog, vector = build(main, handler)
        # b0=BEQ, b1=pc3：STORE(pc4) 在 MEM 级被冲刷
        pipe, seq = run_both(prog, memory_from_words([0, 0]), {1}, vector)

        self.assertEqual(pipe.irq_events[0].saved_pc, 4)
        # 处理器读到旧值 0（被冲刷的 STORE 没有写）；IRET 后 STORE 重执行写 111
        self.assertEqual(pipe.regs[6], 0)
        self.assertEqual(words_from_memory(pipe.memory), [111, 0])
        assert_same_arch_state(self, pipe, seq)


class TestFaultPriority(unittest.TestCase):
    def test_fault_and_interrupt_same_boundary(self):
        handler = [Instruction.iret()]   # pc1
        main = [
            Instruction.addi(1, 0, 3),   # pc2 非对齐基址
            Instruction.load(2, 1, 0),   # pc3 非法 LOAD -> WB 报 ADDR
            Instruction.store(2, 0, 0),  # pc4 年轻 STORE，禁止写
            Instruction.halt(),          # pc5
        ]
        prog, vector = build(main, handler)
        # b0=BEQ, b1=pc2, b2=pc3(故障边界)：脉冲恰好落在故障边界
        pipe, seq = run_both(prog, memory_from_words([9, 9]), {2}, vector)

        # 异常优先：报 ADDR、无接纳事件；中断仍待处理
        self.assertEqual(pipe.fault, Fault.ADDR)
        self.assertEqual(pipe.fault_pc, 3)
        self.assertEqual(pipe.irq_events, ())
        self.assertTrue(pipe.irq_pending)
        self.assertEqual(words_from_memory(pipe.memory), [9, 9])
        self.assertEqual(seq.fault, Fault.ADDR)
        self.assertEqual(seq.fault_pc, 3)
        self.assertTrue(seq.irq_pending)
        assert_same_arch_state(self, pipe, seq)


class TestMaskingAndNesting(unittest.TestCase):
    def test_masked_until_iret_then_pending_accepted(self):
        handler = [
            Instruction.addi(4, 0, 1),   # pc1
            Instruction.addi(4, 4, 1),   # pc2
            Instruction.iret(),          # pc3
        ]
        main = [
            Instruction.addi(1, 0, 7),   # pc4
            Instruction.addi(2, 0, 8),   # pc5
            Instruction.add(3, 1, 2),    # pc6
            Instruction.halt(),          # pc7
        ]
        prog, vector = build(main, handler)
        # b0=BEQ, b1=pc4(接纳#1)；b2/b3 在处理器内（屏蔽中，两个脉冲合并成 1 位）
        pipe, seq = run_both(prog, memory_from_words([0]), {1, 2, 3}, vector)

        # 恰好两次接纳：b1 一次；IRET(b4)解除屏蔽后，b5(pc5 退休)再一次
        self.assertEqual(len(pipe.irq_events), 2)
        self.assertEqual(
            [(e.boundary, e.saved_pc) for e in pipe.irq_events],
            [(1, 5), (5, 6)],
        )
        self.assertEqual(pipe.regs[4], 2)   # 处理器完整跑了两遍
        self.assertEqual(pipe.regs[3], 15)
        self.assertFalse(pipe.irq_pending)  # 合并后的 1 位已被第二次接纳消费
        self.assertFalse(pipe.irq_masked)
        assert_same_arch_state(self, pipe, seq)


class TestHaltBoundary(unittest.TestCase):
    def test_no_acceptance_after_halt(self):
        main = [Instruction.addi(1, 0, 1), Instruction.halt()]
        prog, vector = build(main, [Instruction.iret()])
        # b0=BEQ, b1=pc2, b2=HALT：脉冲落在 HALT 退休边界
        pipe, seq = run_both(prog, memory_from_words([0]), {2}, vector)

        self.assertTrue(pipe.halted)
        self.assertEqual(pipe.irq_events, ())   # HALT 退休后不再接纳
        self.assertTrue(pipe.irq_pending)       # 请求保持，可在结果中观察
        self.assertEqual(pipe.regs[1], 1)
        assert_same_arch_state(self, pipe, seq)


class TestSpuriousIret(unittest.TestCase):
    def test_iret_without_context_is_noop(self):
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.iret(),           # 无中断上下文：空操作
            Instruction.addi(2, 0, 2),
            Instruction.halt(),
        ]
        pipe = run(prog, memory_from_words([0]))
        seq = run_sequential(prog, memory_from_words([0]))
        self.assertEqual([t.pc for t in pipe.trace], [0, 1, 2, 3])
        self.assertEqual((pipe.regs[1], pipe.regs[2]), (1, 2))
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual([t.pc for t in seq.trace], [0, 1, 2, 3])


class TestIllegalVector(unittest.TestCase):
    def setUp(self):
        main = [Instruction.addi(1, 0, 1), Instruction.halt()]
        self.prog, self.vector = build(main, [Instruction.iret()])
        self.n = len(self.prog)

    def test_out_of_range_vector_rejected_before_injection(self):
        for bad in (-1, self.n, self.n + 100):
            mem = bytearray(memory_from_words([5]))
            ctrl = InterruptController(pulse_boundaries={0})
            with self.assertRaises(ProgramError):
                run(self.prog, mem, interrupts=ctrl, interrupt_vector=bad)
            # 拒绝注入：内存未被触碰，请求仍未被消费
            self.assertEqual(list(mem), list(memory_from_words([5])))
            self.assertFalse(ctrl.pending)

    def test_non_integer_vector_rejected(self):
        for bad in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                run(self.prog, memory_from_words([0]),
                    interrupts=InterruptController(), interrupt_vector=bad)

    def test_controller_and_vector_must_come_together(self):
        with self.assertRaises(ProgramError):
            run(self.prog, memory_from_words([0]),
                interrupts=InterruptController())
        with self.assertRaises(ProgramError):
            run(self.prog, memory_from_words([0]), interrupt_vector=1)
        with self.assertRaises(TypeError):
            run(self.prog, memory_from_words([0]),
                interrupts=object(), interrupt_vector=1)

    def test_sequential_same_rejection(self):
        with self.assertRaises(ProgramError):
            run_sequential(self.prog, memory_from_words([0]),
                           interrupts=InterruptController(),
                           interrupt_vector=self.n)


class TestControllerValidation(unittest.TestCase):
    def test_bad_pulse_boundaries(self):
        with self.assertRaises(ValueError):
            InterruptController(pulse_boundaries=[-1])
        with self.assertRaises(TypeError):
            InterruptController(pulse_boundaries=[True])
        with self.assertRaises(TypeError):
            InterruptController(pulse_boundaries=[1.5])
        with self.assertRaises(TypeError):
            InterruptController(pulse_boundaries="3")


class TestInterruptFuzz(unittest.TestCase):
    """带中断的确定性随机程序差分：流水线 vs 独立顺序参考。

    主程序复用 test_equivalence 的随机生成器（只有向前分支，保证终止），
    前面拼一个随机处理器（以 IRET 结尾）；脉冲边界随机。两侧用同一
    边界编号注入，寄存器/内存/退休序列/中断事件必须逐位一致。
    """

    def test_random_programs_with_interrupts(self):
        from test_equivalence import make_program

        for seed in range(200):
            rng = random.Random(10_000 + seed)
            main, init = make_program(seed, length=6 + seed % 10)
            # 随机处理器：1-4 条算术（不写 r0 以外约束）+ IRET
            handler = []
            for _ in range(1 + seed % 4):
                handler.append(
                    Instruction.addi(rng.randrange(1, 32), rng.randrange(32),
                                     rng.randrange(-8, 9))
                )
            handler.append(Instruction.iret())
            prog, vector = build(main, handler)
            pulses = {rng.randrange(0, 40) for _ in range(seed % 4)}
            mem = memory_from_words([(seed * 5 + i * 11) & 0xFFFFFFFF
                                     for i in range(8)])

            pipe = run(prog, bytearray(mem), initial_regs=init,
                       interrupts=InterruptController(pulses),
                       interrupt_vector=vector)
            seq = run_sequential(prog, bytearray(mem), initial_regs=init,
                                 interrupts=InterruptController(pulses),
                                 interrupt_vector=vector)
            assert_same_arch_state(self, pipe, seq)
            self.assertEqual(pipe.fault, seq.fault, seed)
            self.assertEqual(pipe.fault_pc, seq.fault_pc, seed)
            self.assertEqual(pipe.halted, seq.halted, seed)


if __name__ == "__main__":
    unittest.main()
