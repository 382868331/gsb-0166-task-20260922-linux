"""可返回外部中断的精确边界、仲裁和请求保留策略测试。"""

import unittest

from tinycpu import (
    Fault,
    Instruction,
    InterruptContext,
    InterruptRequest,
    memory_from_words,
    words_from_memory,
)
from tinycpu.pipeline import run


class RecordingHandler:
    def __init__(self, service_cycles=2, code_effects=None):
        self.service_cycles = service_cycles
        self.code_effects = code_effects or {}
        self.contexts = []

    def __call__(self, ctx: InterruptContext):
        self.contexts.append(ctx)
        effect = self.code_effects.get(ctx.code, ctx.code + 100)
        ctx.regs[20] = effect
        ctx.memory[0:4] = effect.to_bytes(4, "little")
        return self.service_cycles


class TestExternalInterrupts(unittest.TestCase):
    def test_interrupt_enters_only_after_wb_commit_and_resumes_once(self):
        prog = [
            Instruction.addi(1, 0, 11),
            Instruction.addi(2, 1, 22),
            Instruction.halt(),
        ]
        handler = RecordingHandler(service_cycles=2)
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[InterruptRequest(1, code=7)],
            interrupt_handler=handler,
        )

        self.assertTrue(result.halted)
        self.assertEqual([t.pc for t in result.trace], [0, 1, 2])
        self.assertEqual(len(handler.contexts), 1)
        ctx = handler.contexts[0]
        self.assertEqual((ctx.committed_pc, ctx.resume_pc), (0, 1))
        self.assertEqual(ctx.regs[1], 11)

        record = result.interrupts[0]
        self.assertEqual(record.status, "ENTERED")
        self.assertEqual((record.entry_cycle, record.service_cycles), (5, 2))
        self.assertEqual(record.resume_cycle, 7)
        self.assertEqual(record.committed_pc, 0)
        self.assertEqual(record.resume_pc, 1)
        self.assertEqual(
            [s.reason for c in result.cycles[4:6] for s in c.slots],
            ["iservice"] * 10,
        )
        self.assertEqual(result.regs[1], 11)
        self.assertEqual(result.regs[2], 33)
        self.assertEqual(result.regs[20], 107)
        self.assertEqual(words_from_memory(result.memory)[0], 107)

    def test_resume_pc_uses_branch_decision_from_wb_boundary(self):
        prog = [
            Instruction.addi(1, 0, 1),                 # 0
            Instruction.beq(1, 1, 2),                 # 1 命中 -> 3
            Instruction.addi(2, 0, 99),                # 2 冲刷且中断后不得补执行
            Instruction.addi(3, 0, 7),                 # 3
            Instruction.halt(),                        # 4
        ]
        handler = RecordingHandler(service_cycles=1)
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[InterruptRequest(6)],
            interrupt_handler=handler,
        )

        self.assertTrue(result.halted)
        self.assertEqual(handler.contexts[0].committed_pc, 1)
        self.assertEqual(handler.contexts[0].resume_pc, 3)
        self.assertEqual(result.interrupts[0].resume_pc, 3)
        self.assertEqual([t.pc for t in result.trace], [0, 1, 3, 4])
        self.assertEqual(result.regs[2], 0)
        self.assertEqual(result.regs[3], 7)

    def test_memory_fault_has_priority_over_pending_interrupt(self):
        prog = [
            Instruction.addi(1, 0, 2),
            Instruction.load(2, 1, 0),
            Instruction.halt(),
        ]
        handler = RecordingHandler()
        result = run(
            prog,
            memory_from_words([0, 0]),
            interrupts=[InterruptRequest(6)],
            interrupt_handler=handler,
        )

        self.assertEqual(result.fault, Fault.ADDR)
        self.assertEqual(result.fault_pc, 1)
        self.assertEqual(result.fault_cycle, 6)
        self.assertEqual(result.interrupts, ())
        self.assertIsNotNone(result.pending_interrupt)
        self.assertEqual(handler.contexts, [])

    def test_halt_has_priority_over_interrupt_arriving_at_same_wb(self):
        prog = [Instruction.addi(1, 0, 1), Instruction.halt()]
        handler = RecordingHandler()
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[InterruptRequest(6)],
            interrupt_handler=handler,
        )

        self.assertTrue(result.halted)
        self.assertEqual(result.interrupts, ())
        self.assertIsNotNone(result.pending_interrupt)
        self.assertEqual(handler.contexts, [])

    def test_wb_fault_beats_nearby_branch_and_pending_interrupt(self):
        prog = [
            Instruction.addi(1, 0, 2),                 # 0
            Instruction.load(2, 1, 0),                 # 1 非对齐，WB 报故障
            Instruction.beq(0, 0, 2),                  # 2 同周期在 MEM，必须丢弃
            Instruction.addi(3, 0, 99),                # 3 分支目标，绝不执行
            Instruction.halt(),                        # 4
        ]
        handler = RecordingHandler()
        result = run(
            prog,
            memory_from_words([0, 0]),
            interrupts=[InterruptRequest(6, code=40)],
            interrupt_handler=handler,
        )

        self.assertEqual((result.fault, result.fault_pc, result.fault_cycle),
                         (Fault.ADDR, 1, 6))
        self.assertEqual([t.pc for t in result.trace], [0, 1])
        self.assertEqual(result.regs[3], 0)
        self.assertEqual(result.pending_interrupt.code, 40)
        self.assertEqual(handler.contexts, [])

    def test_busy_default_rejects_third_request_without_silent_loss(self):
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(2, 0, 2),
            Instruction.addi(3, 0, 3),
            Instruction.addi(4, 0, 4),
            Instruction.addi(5, 0, 5),
            Instruction.addi(6, 0, 6),
            Instruction.halt(),
        ]
        handler = RecordingHandler(
            service_cycles=3,
            code_effects={10: 110, 20: 120},
        )
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[
                InterruptRequest(1, 10),
                InterruptRequest(7, 20),
                InterruptRequest(8, 30),
            ],
            interrupt_handler=handler,
        )

        self.assertEqual([i.code for i in result.interrupts], [10, 20])
        self.assertEqual([i.code for i in result.rejected_interrupts], [30])
        self.assertEqual([c.code for c in handler.contexts], [10, 20])
        self.assertEqual([t.pc for t in result.trace], list(range(7)))

    def test_busy_replace_keeps_latest_and_records_replaced_pending(self):
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(2, 0, 2),
            Instruction.addi(3, 0, 3),
            Instruction.addi(4, 0, 4),
            Instruction.addi(5, 0, 5),
            Instruction.addi(6, 0, 6),
            Instruction.halt(),
        ]
        handler = RecordingHandler(
            service_cycles=3,
            code_effects={10: 110, 30: 130},
        )
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[
                InterruptRequest(1, 10),
                InterruptRequest(7, 20),
                InterruptRequest(8, 30),
            ],
            interrupt_handler=handler,
            interrupt_on_busy="replace",
        )

        self.assertEqual([i.code for i in result.interrupts], [10, 30])
        self.assertEqual([i.code for i in result.rejected_interrupts], [20])
        self.assertEqual([c.code for c in handler.contexts], [10, 30])

    def test_duplicate_request_before_service_is_rejected(self):
        prog = [Instruction.addi(1, 0, 1), Instruction.halt()]
        handler = RecordingHandler(service_cycles=1)
        result = run(
            prog,
            memory_from_words([0]),
            interrupts=[InterruptRequest(1, 1), InterruptRequest(1, 2)],
            interrupt_handler=handler,
        )

        self.assertEqual([i.code for i in result.interrupts], [1])
        self.assertEqual([i.code for i in result.rejected_interrupts], [2])

    def test_interrupt_inputs_are_validated(self):
        prog = [Instruction.halt()]
        with self.assertRaisesRegex(TypeError, "InterruptRequest"):
            run(prog, memory_from_words([0]), interrupts=[object()])
        with self.assertRaisesRegex(ValueError, r"\[1, 2000\]"):
            run(
                prog,
                memory_from_words([0]),
                interrupts=[InterruptRequest(0)],
            )
        with self.assertRaisesRegex(ValueError, "32 位"):
            run(
                prog,
                memory_from_words([0]),
                interrupts=[InterruptRequest(1, 1 << 32)],
            )
        with self.assertRaisesRegex(TypeError, "int"):
            run(
                prog,
                memory_from_words([0]),
                interrupts=[InterruptRequest(1, True)],
            )
        with self.assertRaisesRegex(TypeError, "int"):
            run(
                prog,
                memory_from_words([0]),
                interrupt_service_cycles=True,
            )
        with self.assertRaisesRegex(ValueError, r"\[1, 2000\]"):
            run(
                prog,
                memory_from_words([0]),
                interrupt_service_cycles=0,
            )
        with self.assertRaisesRegex(ValueError, "reject"):
            run(
                prog,
                memory_from_words([0]),
                interrupt_on_busy="drop",
            )


if __name__ == "__main__":
    unittest.main()
