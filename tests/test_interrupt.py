"""可返回外部中断的针对性测试与差分验证。

覆盖：
* 中断只在 WB 提交后的指令边界进入（精确现场），IRET 从断点继续；
* 不重复提交、不遗漏指令效果（主程序退休序列与无中断基线完全一致）；
* 同周期优先级：WB 故障 > HALT 退休 > 中断采纳（分支改向被中断覆盖，
  分支返回后重新执行，效果不重复）；
* handler 内不嵌套：期间到达的请求按电平语义保留，IRET 退休后立即补服务；
* 运行结束时仍未服务的请求在 interrupts_pending 中报告，不静默丢失；
* 与独立顺序参考解释器在相同合成程序 + 本地模拟时间下交叉验证。
"""

import unittest

from tinycpu import (
    Fault,
    Instruction,
    InterruptRequest,
    ProgramError,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

from test_equivalence import make_program, MEM_WORDS


def _counting_handler() -> list[Instruction]:
    """最小处理程序：r31 += 1 后 IRET 返回。"""
    return [Instruction.addi(31, 31, 1), Instruction.iret()]


def _main_pcs(result) -> list[int]:
    return [t.pc for t in result.trace if t.space == "main"]


def _reserve_r31(prog, init):
    """把主程序里的 r31 全部改写到 r30，使 r31 成为 handler 专用寄存器。"""
    def map(ins):
        return Instruction(
            ins.op,
            rd=30 if ins.rd == 31 else ins.rd,
            rs1=30 if ins.rs1 == 31 else ins.rs1,
            rs2=30 if ins.rs2 == 31 else ins.rs2,
            imm=ins.imm,
        )
    new_init = list(init)
    new_init[31] = 0
    return [map(i) for i in prog], new_init


class TestInterruptBasic(unittest.TestCase):
    def test_take_and_return_resume_exactly_once(self):
        prog = [
            Instruction.addi(1, 0, 10),   # pc0
            Instruction.addi(2, 0, 20),   # pc1
            Instruction.addi(3, 0, 30),   # pc2
            Instruction.addi(4, 0, 40),   # pc3
            Instruction.halt(),           # pc4
        ]
        irq = [InterruptRequest(3)]
        r = run(prog, memory_from_words([0] * 4), interrupts=irq,
                handler=_counting_handler())
        self.assertTrue(r.halted)
        self.assertEqual(r.interrupts_taken, 1)
        self.assertEqual(r.interrupts_pending, 0)
        # 主程序每条指令恰好提交一次：不重复、不遗漏。
        self.assertEqual(_main_pcs(r), [0, 1, 2, 3, 4])
        self.assertEqual(r.regs[1:5], (10, 20, 30, 40))
        self.assertEqual(r.regs[31], 1)  # handler 效果恰好一次
        # take 保存的断点与 return 恢复的断点一致。
        takes = [x for x in r.interrupt_trace if x.kind == "take"]
        rets = [x for x in r.interrupt_trace if x.kind == "return"]
        self.assertEqual(len(takes), len(rets), 1)
        self.assertEqual(takes[0].pc, rets[0].pc)
        # 断点必须是最老未提交指令：take 之后退休的主程序指令
        # 恰好是断点开始的剩余序列。
        self.assertEqual(_main_pcs(r)[takes[0].pc:],
                         list(range(takes[0].pc, 5)))

    def test_resume_point_is_oldest_uncommitted(self):
        # 周期 5：pc0 在 WB 提交，pc1 在 MEM —— 断点必须是 pc1，
        # pc0 不得重复提交。
        prog = [Instruction.addi(i + 1, 0, i + 1) for i in range(6)]
        prog.append(Instruction.halt())
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(5)], handler=_counting_handler())
        take = next(x for x in r.interrupt_trace if x.kind == "take")
        self.assertEqual(take.pc, 1)
        self.assertEqual(_main_pcs(r), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(r.regs[1:7], (1, 2, 3, 4, 5, 6))

    def test_store_in_flight_commits_exactly_once(self):
        # 中断采纳时 MEM 级有一条 STORE：它未提交，返回后重新执行，
        # 内存必须只被写一次（值正确即可，且轨迹中只出现一次）。
        prog = [
            Instruction.addi(1, 0, 4),     # pc0 基址 4
            Instruction.addi(2, 0, 99),    # pc1
            Instruction.store(2, 1, 0),    # pc2  mem[4] <- 99
            Instruction.halt(),            # pc3
        ]
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(4)], handler=_counting_handler())
        stores = [t for t in r.trace if t.space == "main" and t.addr is not None]
        self.assertEqual(len(stores), 1)
        self.assertEqual(words_from_memory(r.memory)[1], 99)

    def test_irq_slot_marks_flushed_young_instructions(self):
        prog = [Instruction.addi(1, 0, 1), Instruction.addi(2, 0, 2),
                Instruction.addi(3, 0, 3), Instruction.halt()]
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(3)], handler=_counting_handler())
        take_cycle = next(x.cycle for x in r.interrupt_trace if x.kind == "take")
        slots = r.cycles[take_cycle - 1].slots
        irq_stages = [s.stage for s in slots if s.reason == "irq"]
        self.assertTrue(irq_stages)  # 被冲刷的年轻指令在槽位中可见


class TestInterruptPriority(unittest.TestCase):
    def test_fault_beats_interrupt_same_cycle(self):
        # 非法 STORE 与中断请求在同一 WB 周期相遇：故障优先，
        # 机器停在故障点，中断保持未决并被报告（不静默丢失）。
        prog = [
            Instruction.addi(1, 0, 5),      # pc0 非对齐基址
            Instruction.store(1, 1, 0),     # pc1 非法 STORE
            Instruction.halt(),             # pc2
        ]
        # pc1 在周期 7 到 WB；请求周期 7 恰好同周期到达。
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(7)], handler=_counting_handler())
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertFalse(r.fault_in_handler)
        self.assertEqual(r.interrupts_taken, 0)
        self.assertEqual(r.interrupts_pending, 1)
        self.assertEqual(words_from_memory(r.memory), [0] * 4)

    def test_halt_beats_interrupt(self):
        # HALT 退休的周期请求才到期：程序已结束，不进入 handler，
        # 请求计入 pending 报告。
        prog = [Instruction.addi(1, 0, 1), Instruction.halt()]
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(6)], handler=_counting_handler())
        self.assertTrue(r.halted)
        self.assertEqual(r.interrupts_taken, 0)
        self.assertEqual(r.interrupts_pending, 1)
        self.assertEqual(r.regs[31], 0)

    def test_interrupt_overrides_branch_redirect_and_reexecutes(self):
        # 分支在 EX 决策改向的同一周期采纳中断：分支是推测性的，
        # 被冲刷后返回重新执行，效果恰好一次。
        prog = [
            Instruction.addi(1, 0, 7),      # pc0
            Instruction.beq(1, 1, 2),       # pc1 恒真 -> pc3
            Instruction.addi(2, 0, 100),    # pc2 （被分支冲刷）
            Instruction.addi(3, 0, 30),     # pc3 分支目标
            Instruction.halt(),             # pc4
        ]
        # pc1 在周期 5 到 EX 决策；请求周期 5 同周期到达。
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(5)], handler=_counting_handler())
        self.assertTrue(r.halted)
        self.assertEqual(_main_pcs(r), [0, 1, 3, 4])  # pc2 不出现，pc1 不重复
        self.assertEqual(r.regs[1], 7)
        self.assertEqual(r.regs[2], 0)   # 被冲刷的 pc2 从未提交
        self.assertEqual(r.regs[3], 30)
        self.assertEqual(r.regs[31], 1)


class TestInterruptNesting(unittest.TestCase):
    def test_request_during_handler_is_retained_not_dropped(self):
        prog = [Instruction.addi(1, 0, 1), Instruction.addi(2, 0, 2),
                Instruction.halt()]
        handler = [Instruction.addi(31, 31, 1),
                   Instruction.addi(31, 31, 1),
                   Instruction.iret()]
        # 第二个请求在 handler 执行期间到达：保留，IRET 后立即补服务。
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(3), InterruptRequest(4)],
                handler=handler)
        self.assertEqual(r.interrupts_taken, 2)
        self.assertEqual(r.interrupts_pending, 0)
        self.assertEqual(r.regs[31], 4)  # 两次 handler，每次 +2
        kinds = [x.kind for x in r.interrupt_trace]
        self.assertEqual(kinds, ["take", "return", "take", "return"])
        # 第二次 take 与第一次 return 同周期（IRET 退休后立即补服务）。
        self.assertEqual(r.interrupt_trace[1].cycle, r.interrupt_trace[2].cycle)
        self.assertEqual(_main_pcs(r), [0, 1, 2])

    def test_many_requests_same_cycle_all_retained(self):
        prog = [Instruction.addi(1, 0, 1), Instruction.halt()]
        r = run(prog, memory_from_words([0] * 4),
                interrupts=[InterruptRequest(2)] * 3, handler=_counting_handler())
        self.assertEqual(r.interrupts_taken, 3)
        self.assertEqual(r.regs[31], 3)
        self.assertEqual(r.interrupts_pending, 0)


class TestInterruptValidation(unittest.TestCase):
    def setUp(self):
        self.prog = [Instruction.addi(1, 0, 1), Instruction.halt()]
        self.mem = memory_from_words([0])

    def test_interrupts_without_handler_rejected(self):
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, interrupts=[InterruptRequest(1)])

    def test_handler_without_interrupts_rejected(self):
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, handler=_counting_handler())

    def test_iret_in_main_program_rejected(self):
        with self.assertRaises(ProgramError):
            run([Instruction.iret(), Instruction.halt()], self.mem)

    def test_handler_must_end_with_iret(self):
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, interrupts=[InterruptRequest(1)],
                handler=[Instruction.addi(1, 0, 1)])

    def test_handler_rejects_halt_and_inner_iret_and_bad_beq(self):
        irq = [InterruptRequest(1)]
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, interrupts=irq,
                handler=[Instruction.halt(), Instruction.iret()])
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, interrupts=irq,
                handler=[Instruction.iret(), Instruction.iret()])
        with self.assertRaises(ProgramError):
            run(self.prog, self.mem, interrupts=irq,
                handler=[Instruction.beq(1, 1, 5), Instruction.iret()])

    def test_bad_request_cycle_rejected(self):
        for bad in (0, -1, 2001, 2.5, float("nan"), True):
            with self.assertRaises((TypeError, ValueError)):
                InterruptRequest(bad)

    def test_wrong_request_type_rejected(self):
        with self.assertRaises(TypeError):
            run(self.prog, self.mem, interrupts=[3],
                handler=_counting_handler())


class TestInterruptEquivalence(unittest.TestCase):
    """与独立顺序参考在合成程序 + 本地模拟时间下的交叉验证。

    两种模拟时间域（流水线周期 / 顺序退休步数）都会把请求服务完，
    因此 handler 效果（r31 += 请求数）与最终架构状态必须一致；
    主程序退休序列必须与无中断基线完全一致（不重复、不遗漏）。
    """

    def test_random_programs_with_interrupts(self):
        for seed in range(120):
            prog, init = make_program(seed)
            prog, init = _reserve_r31(prog, init)
            mem = memory_from_words([0] * MEM_WORDS)
            irq = [InterruptRequest(1), InterruptRequest(2), InterruptRequest(3)]
            handler = _counting_handler()

            base = run(prog, mem, init)
            pipe = run(prog, mem, init,
                       interrupts=irq, handler=handler)
            seq = run_sequential(prog, mem, init,
                                 interrupts=irq, handler=handler)

            with self.subTest(seed=seed):
                if base.fault is not None:
                    # 基线出异常：故障前可能已服务部分请求；只检查
                    # 故障一致性与"请求不静默丢失"。
                    self.assertEqual(pipe.fault, seq.fault)
                    self.assertEqual(pipe.fault_pc, seq.fault_pc)
                    self.assertEqual(pipe.fault_in_handler, False)
                    self.assertEqual(seq.fault_in_handler, False)
                    self.assertEqual(
                        pipe.interrupts_taken + pipe.interrupts_pending, 3)
                    self.assertEqual(
                        seq.interrupts_taken + seq.interrupts_pending, 3)
                    continue
                # 正常结束：所有请求都应在两种时间域内服务完。
                self.assertEqual(pipe.interrupts_taken, 3)
                self.assertEqual(seq.interrupts_taken, 3)
                self.assertEqual(pipe.interrupts_pending, 0)
                self.assertEqual(seq.interrupts_pending, 0)
                # 主程序退休序列与无中断基线完全一致。
                self.assertEqual(_main_pcs(pipe), [t.pc for t in base.trace])
                # 最终架构状态：r31 恰好多了 3 次 handler 增量，
                # 其余寄存器与内存与基线一致；流水线与顺序机一致。
                self.assertEqual(pipe.regs, seq.regs)
                self.assertEqual(pipe.memory, seq.memory)
                self.assertEqual(pipe.memory, base.memory)
                for r in range(31):
                    self.assertEqual(pipe.regs[r], base.regs[r])
                self.assertEqual(pipe.regs[31],
                                 (base.regs[31] + 3) & 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
