"""输入校验与错误语义测试。"""

import math
import unittest

from tinycpu import (
    ADDI,
    HALT,
    LOAD,
    MAX_INSTRUCTIONS,
    Instruction,
    ProgramError,
    memory_from_words,
)
from tinycpu.isa import check_int
from tinycpu.pipeline import run


class TestScalarChecks(unittest.TestCase):
    def test_rejects_bool_as_int(self):
        with self.assertRaisesRegex(TypeError, "bool"):
            check_int("x", True)
        with self.assertRaisesRegex(TypeError, "bool"):
            check_int("x", False)

    def test_rejects_non_finite_float(self):
        for fv in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                check_int("x", fv)

    def test_rejects_finite_float(self):
        with self.assertRaisesRegex(TypeError, "float"):
            check_int("x", 1.5)

    def test_accepts_plain_int(self):
        self.assertEqual(check_int("x", -3), -3)
        self.assertEqual(check_int("x", 0), 0)


class TestInstructionValidation(unittest.TestCase):
    def test_register_range(self):
        with self.assertRaisesRegex(ValueError, "rd"):
            Instruction.add(32, 1, 2)
        with self.assertRaisesRegex(ValueError, "rs1"):
            Instruction.add(1, -1, 2)
        # bool 不能冒充寄存器号
        with self.assertRaisesRegex(TypeError, "bool"):
            Instruction.addi(True, 1, 2)  # type: ignore[arg-type]

    def test_immediate_range(self):
        Instruction.addi(1, 1, (1 << 31) - 1)
        Instruction.addi(1, 1, -(1 << 31))
        with self.assertRaises(ValueError):
            Instruction.addi(1, 1, 1 << 31)
        with self.assertRaises(ValueError):
            Instruction.addi(1, 1, -(1 << 31) - 1)
        with self.assertRaises(ValueError):
            Instruction.addi(1, 1, float("nan"))  # type: ignore[arg-type]

    def test_op_must_be_enum(self):
        with self.assertRaises(TypeError):
            Instruction("ADD", rd=1, rs1=2, rs2=3)  # type: ignore[arg-type]

    def test_constructors_round_trip(self):
        self.assertEqual(Instruction.halt().op, HALT)
        self.assertEqual(Instruction.load(1, 2, 8).op, LOAD)
        self.assertEqual(Instruction.beq(1, 2, 3).imm, 3)


class TestProgramValidation(unittest.TestCase):
    def setUp(self):
        self.mem = memory_from_words([0, 0])

    def test_program_must_end_with_halt(self):
        with self.assertRaisesRegex(ProgramError, "HALT"):
            run([Instruction.addi(1, 0, 1)], self.mem)

    def test_program_length_bounds(self):
        with self.assertRaisesRegex(ProgramError, r"\[1, 200\]"):
            run([], self.mem)
        big = [Instruction.addi(1, 0, 0)] * MAX_INSTRUCTIONS + [Instruction.halt()]
        with self.assertRaisesRegex(ProgramError, "200"):
            run(big, self.mem)

    def test_branch_target_in_program(self):
        prog = [
            Instruction.beq(0, 0, 5),  # 目标 0+5 越界
            Instruction.halt(),
        ]
        with self.assertRaisesRegex(ProgramError, "BEQ 目标越界"):
            run(prog, self.mem)
        prog2 = [
            Instruction.beq(0, 0, -1),  # 负目标越界
            Instruction.halt(),
        ]
        with self.assertRaises(ProgramError):
            run(prog2, self.mem)

    def test_memory_must_be_aligned_nonempty_bytes(self):
        with self.assertRaises(ProgramError):
            run([Instruction.halt()], b"")
        with self.assertRaises(ProgramError):
            run([Instruction.halt()], b"abc")
        with self.assertRaises(TypeError):
            run([Instruction.halt()], [0, 0, 0, 0])  # type: ignore[arg-type]

    def test_initial_regs_shape_and_r0(self):
        with self.assertRaisesRegex(ProgramError, "长度必须为 32"):
            run([Instruction.halt()], self.mem, [0] * 31)
        regs = [0] * 32
        regs[0] = 1
        with self.assertRaisesRegex(ProgramError, "r0 恒零"):
            run([Instruction.halt()], self.mem, regs)
        regs2 = [0] * 32
        regs2[1] = 1 << 32
        with self.assertRaises(ValueError):
            run([Instruction.halt()], self.mem, regs2)
        regs3 = [0] * 32
        regs3[1] = True  # type: ignore[list-item]
        with self.assertRaisesRegex(TypeError, "bool"):
            run([Instruction.halt()], self.mem, regs3)

    def test_memory_words_validation(self):
        with self.assertRaisesRegex(TypeError, "int"):
            memory_from_words([True])
        with self.assertRaises(ValueError):
            memory_from_words([1 << 32])
        with self.assertRaises(TypeError):
            memory_from_words("abc")  # type: ignore[arg-type]

    def test_failure_leaves_no_partial_state(self):
        # 非法输入在执行前抛出；调用方传入的内存不被修改。
        mem = bytearray(b"\x01\x02\x03\x04")
        snapshot = bytes(mem)
        with self.assertRaises(Exception):
            run([Instruction.addi(1, 0, float("inf"))], mem)  # type: ignore[arg-type]
        self.assertEqual(bytes(mem), snapshot)

    def test_instructions_elements_typed(self):
        with self.assertRaisesRegex(TypeError, r"instructions\[0\]"):
            run(["addi r1 r0 1", Instruction.halt()], self.mem)  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
