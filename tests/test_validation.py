"""输入契约与错误语义测试。"""

import math
import unittest

from pipeline_cpu import (
    Instruction, Opcode, InvalidProgramError, CycleLimitExceeded,
    run_pipeline, run_sequential,
    MAX_PROGRAM_INSTRUCTIONS, MAX_CYCLES,
)


H = lambda: Instruction(Opcode.HALT)


class ValidationTests(unittest.TestCase):
    def test_bool_rejected_everywhere(self):
        # bool 不得冒充整数（消息带字段定位：rd / imm / max_cycles）。
        with self.assertRaisesRegex(InvalidProgramError, r"ADD\.rd.*bool|bool"):
            run_pipeline([Instruction(Opcode.ADD, a=True, b=0, c=0), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "imm"):
            run_pipeline([Instruction(Opcode.ADDI, a=1, b=0, imm=True), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "max_cycles"):
            run_pipeline([H()], bytes(16), max_cycles=True)

    def test_float_nan_infinity_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, 1.5, 0.0):
            with self.assertRaisesRegex(InvalidProgramError, "必须是 int"):
                run_pipeline(
                    [Instruction(Opcode.ADDI, a=1, b=0, imm=bad), H()],
                    bytes(16))

    def test_register_range(self):
        with self.assertRaisesRegex(InvalidProgramError, "寄存器号越界"):
            run_pipeline([Instruction(Opcode.ADD, a=32, b=0, c=0), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "寄存器号越界"):
            run_pipeline([Instruction(Opcode.ADD, a=-1, b=0, c=0), H()],
                         bytes(16))

    def test_imm16_range(self):
        Instruction(Opcode.ADDI, a=1, b=0, imm=32767)  # 边界合法
        Instruction(Opcode.ADDI, a=1, b=0, imm=-32768)
        with self.assertRaisesRegex(InvalidProgramError, "16 位"):
            run_pipeline([Instruction(Opcode.ADDI, a=1, b=0, imm=32768), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "16 位"):
            run_pipeline(
                [Instruction(Opcode.ADDI, a=1, b=0, imm=-32769), H()],
                bytes(16))

    def test_branch_imm_range_and_alignment(self):
        Instruction(Opcode.BEQ, b=0, c=0, imm=-1048576)
        Instruction(Opcode.BEQ, b=0, c=0, imm=1048572)
        with self.assertRaisesRegex(InvalidProgramError, "21 位"):
            run_pipeline(
                [Instruction(Opcode.BEQ, b=0, c=0, imm=1048576), H()],
                bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "21 位"):
            run_pipeline(
                [Instruction(Opcode.BEQ, b=0, c=0, imm=-1048580), H()],
                bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "对齐"):
            run_pipeline([Instruction(Opcode.BEQ, b=0, c=0, imm=2), H()],
                         bytes(16))

    def test_unused_fields_must_be_zero(self):
        with self.assertRaisesRegex(InvalidProgramError, "imm"):
            run_pipeline([Instruction(Opcode.ADD, a=1, b=0, c=0, imm=1), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "c"):
            run_pipeline([Instruction(Opcode.ADDI, a=1, b=0, c=2, imm=1), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "a"):
            run_pipeline([Instruction(Opcode.STORE, a=1, b=0, c=0), H()],
                         bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "全 0"):
            run_pipeline([Instruction(Opcode.HALT, imm=0, a=1)], bytes(16))

    def test_program_container_rules(self):
        with self.assertRaisesRegex(InvalidProgramError, "list/tuple"):
            run_pipeline(H(), bytes(16))  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidProgramError, "至少包含 1 条"):
            run_pipeline([], bytes(16))
        with self.assertRaisesRegex(InvalidProgramError, "Instruction"):
            run_pipeline([("ADD", 1, 0, 0)], bytes(16))
        too_many = [Instruction(Opcode.ADDI, a=1, b=0, imm=0)] * (
            MAX_PROGRAM_INSTRUCTIONS + 1)
        with self.assertRaisesRegex(InvalidProgramError, "上限"):
            run_pipeline(too_many, bytes(16))

    def test_program_size_at_limit_accepted(self):
        prog = [Instruction(Opcode.ADDI, a=1, b=0, imm=0)] * (
            MAX_PROGRAM_INSTRUCTIONS - 1) + [H()]
        r = run_pipeline(prog, bytes(16))
        self.assertTrue(r.halted)

    def test_memory_rules(self):
        with self.assertRaisesRegex(InvalidProgramError, "bytes/bytearray"):
            run_pipeline([H()], [0, 0, 0, 0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidProgramError, "对齐"):
            run_pipeline([H()], b"abc")
        big = bytes(4 * ((1 << 28) + 1))
        with self.assertRaisesRegex(InvalidProgramError, "超过上限"):
            run_pipeline([H()], big)

    def test_bad_opcode_enum(self):
        with self.assertRaises(InvalidProgramError):
            Instruction(opcode="ADD", a=1)  # type: ignore[arg-type]

    def test_caller_memory_not_mutated_and_cycle_budget(self):
        src = bytearray((99).to_bytes(4, "little"))
        snapshot = bytes(src)
        run_pipeline(
            [Instruction(Opcode.STORE, b=0, c=0, imm=0), H()], src)
        self.assertEqual(bytes(src), snapshot)  # 输入内存不被修改

        with self.assertRaisesRegex(InvalidProgramError, "max_cycles"):
            run_pipeline([H()], bytes(16), max_cycles=0)
        with self.assertRaisesRegex(InvalidProgramError, "max_cycles"):
            run_pipeline([H()], bytes(16), max_cycles=MAX_CYCLES + 1)

    def test_cycle_budget_exhaustion_is_distinct_from_halt(self):
        # 预算耗尽是独立错误，绝不与“停机/无解”混淆。
        prog = [Instruction(Opcode.ADDI, a=1, b=0, imm=1), H()]
        with self.assertRaises(CycleLimitExceeded):
            run_pipeline(prog, bytes(16), max_cycles=3)

    def test_sequential_shares_contract(self):
        with self.assertRaisesRegex(InvalidProgramError, "寄存器号越界"):
            run_sequential([Instruction(Opcode.ADD, a=99, b=0, c=0)],
                           bytes(16))


if __name__ == "__main__":
    unittest.main()
