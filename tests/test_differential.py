"""差分测试：流水线结果必须与独立顺序解释器逐项一致。

顺序参考解释器不调用被测核心（见 pipeline_cpu/reference.py），因此
这里的比对是两个独立实现之间的交叉验证，而非往返一致。
"""

import random
import unittest

from pipeline_cpu import (
    Instruction, Opcode, run_pipeline, run_sequential,
)


def words_to_bytes(words):
    return b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words)


class DifferentialSingleCaseTests(unittest.TestCase):
    def _assert_same_arch_state(self, prog, mem, msg=""):
        p = run_pipeline(prog, mem)
        s = run_sequential(prog, mem)
        self.assertEqual(p.halted, s.halted, msg)
        self.assertEqual(
            (p.fault.kind if p.fault else None,
             p.fault.address if p.fault else None),
            (s.fault.kind if s.fault else None,
             s.fault.address if s.fault else None),
            msg,
        )
        self.assertEqual(p.final_registers, s.final_registers, msg)
        self.assertEqual(p.final_memory, s.final_memory, msg)
        # 退休值序列：流水线非故障退休项的 (seq, rd, value) 与顺序执行
        # 轨迹的前三字段必须一致（r0 写回两边都记 rd=0/value=None）。
        p_trail = [(r.seq, r.rd, r.value)
                   for r in p.retirements if r.fault is None]
        s_trail = [(seq, rd, val)
                   for seq, rd, val, _h in s.executed][:len(p_trail)]
        self.assertEqual(p_trail, s_trail, msg)
        # 故障指令的静态序号一致（顺序模型给 fault_seq）。
        if s.fault is not None and s.fault.kind == "bad_address":
            self.assertEqual(p.fault_seq, s.fault_seq, msg)
        return p, s

    def test_arithmetic_chain(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=10),
            Instruction(Opcode.ADDI, a=2, b=1, imm=20),
            Instruction(Opcode.ADD, a=3, b=1, c=2),
            Instruction(Opcode.SUB, a=4, b=3, c=1),
            Instruction(Opcode.HALT),
        ]
        p, _ = self._assert_same_arch_state(prog, bytes(16))
        self.assertEqual(p.final_registers[3], 40)
        self.assertEqual(p.final_registers[4], 30)

    def test_memory_round_trip_with_stalls(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=8),
            Instruction(Opcode.ADDI, a=2, b=0, imm=123),
            Instruction(Opcode.STORE, b=1, c=2, imm=0),  # mem[8]=123
            Instruction(Opcode.LOAD, a=3, b=1, imm=0),   # r3=123 (无冒险)
            Instruction(Opcode.LOAD, a=4, b=0, imm=8),   # r4=123
            Instruction(Opcode.ADD, a=5, b=3, c=4),      # load-use(seq4)
            Instruction(Opcode.HALT),
        ]
        mem = words_to_bytes([0, 0, 0, 0, 0])
        p, _ = self._assert_same_arch_state(prog, mem)
        self.assertEqual(p.final_registers[5], 246)
        self.assertEqual(len([t for t in p.trace if "stall" in t.note]), 1)

    def test_branch_heavy(self):
        # 0: r1=0
        # 1: r1+=1
        # 2: r2=5
        # 3: beq r1,r2 -> seq5(+8) 退出
        # 4: beq r0,r0 -> seq1(-12)
        # 5: store r1 -> mem[0] = 5
        # 6: halt
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=0),
            Instruction(Opcode.ADDI, a=1, b=1, imm=1),
            Instruction(Opcode.ADDI, a=2, b=0, imm=5),
            Instruction(Opcode.BEQ, b=1, c=2, imm=8),
            Instruction(Opcode.BEQ, b=0, c=0, imm=-12),
            Instruction(Opcode.STORE, b=0, c=1, imm=0),
            Instruction(Opcode.HALT),
        ]
        p, _ = self._assert_same_arch_state(prog, words_to_bytes([0]))
        self.assertEqual(p.final_memory, words_to_bytes([5]))

    def test_fault_programs(self):
        cases = [
            # 非法 LOAD，前面有合法 STORE
            ([Instruction(Opcode.ADDI, a=1, b=0, imm=7),
              Instruction(Opcode.STORE, b=0, c=1, imm=0),
              Instruction(Opcode.LOAD, a=2, b=0, imm=100),
              Instruction(Opcode.HALT)],
             words_to_bytes([0])),
            # 未对齐地址
            ([Instruction(Opcode.ADDI, a=1, b=0, imm=2),
              Instruction(Opcode.LOAD, a=2, b=1, imm=1),
              Instruction(Opcode.HALT)],
             words_to_bytes([0, 0])),
            # 环绕地址：r1=0xFFFFFFFC, +8 => 4 (wrap) 合法 vs +12 => 8
            ([Instruction(Opcode.LOAD, a=1, b=0, imm=-4),
              Instruction(Opcode.HALT)],
             words_to_bytes([0, 0])),
        ]
        for prog, mem in cases:
            with self.subTest(prog=prog):
                self._assert_same_arch_state(prog, mem)

    def test_negative_immediate_address_wrap(self):
        # 负偏移相对基址寻址：r1=12(内存末端), LOAD [r1-4] 读到最后一个字。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=12),
            Instruction(Opcode.LOAD, a=2, b=1, imm=-4),
            Instruction(Opcode.HALT),
        ]
        mem = words_to_bytes([11, 22, 33])
        p, s = self._assert_same_arch_state(prog, mem)
        self.assertEqual(p.final_registers[2], 33)

    def test_32bit_address_wrap_stays_out_of_range(self):
        # r0=0 + imm=-4 => 0xFFFFFFFC，对任何实际内存都越界而非环绕命中。
        prog = [
            Instruction(Opcode.LOAD, a=1, b=0, imm=-4),
            Instruction(Opcode.HALT),
        ]
        mem = words_to_bytes([11, 22, 33])
        p, s = self._assert_same_arch_state(prog, mem)
        self.assertEqual(p.fault.kind, "bad_address")
        self.assertEqual(p.fault.address, 0xFFFFFFFC)


class DifferentialRandomTests(unittest.TestCase):
    """确定性种子的随机程序：保证终止（仅前向 BEQ），含冒险与访存。"""

    def _gen_program(self, rng, n):
        prog = []
        # 先放数据初始化，再随机主体，最后 HALT。
        for i in range(n):
            kind = rng.randrange(7)
            if kind == 0:
                prog.append(Instruction(
                    Opcode.ADD, a=rng.randrange(1, 16),
                    b=rng.randrange(0, 16), c=rng.randrange(0, 16)))
            elif kind == 1:
                prog.append(Instruction(
                    Opcode.SUB, a=rng.randrange(1, 16),
                    b=rng.randrange(0, 16), c=rng.randrange(0, 16)))
            elif kind == 2:
                prog.append(Instruction(
                    Opcode.ADDI, a=rng.randrange(1, 16),
                    b=rng.randrange(0, 16), imm=rng.randrange(-100, 100)))
            elif kind == 3:
                prog.append(Instruction(
                    Opcode.LOAD, a=rng.randrange(1, 16),
                    b=rng.randrange(0, 4), imm=4 * rng.randrange(0, 8)))
            elif kind == 4:
                prog.append(Instruction(
                    Opcode.STORE, b=rng.randrange(0, 4),
                    c=rng.randrange(0, 16), imm=4 * rng.randrange(0, 8)))
            elif kind == 5:
                # 仅前向跳转，偏移对齐且不越过程序尾（目标 <= n-1）。
                max_delta = (n - 1 - i) * 4
                if max_delta >= 8:
                    imm = 4 * rng.randrange(2, max_delta // 4 + 1)
                    prog.append(Instruction(
                        Opcode.BEQ, b=rng.randrange(0, 16),
                        c=rng.randrange(0, 16), imm=imm))
                else:
                    prog.append(Instruction(
                        Opcode.ADDI, a=rng.randrange(1, 16),
                        b=0, imm=rng.randrange(-50, 50)))
            else:
                prog.append(Instruction(
                    Opcode.ADDI, a=rng.randrange(1, 16),
                    b=rng.randrange(0, 16), imm=rng.randrange(-100, 100)))
        # base 寄存器 r0..r3 给小范围内的合法字偏移，制造稀疏非法访问。
        # 不主动限制：r0=0 固定，其它基址可能为任意值 => 自然产生故障。
        prog.append(Instruction(Opcode.HALT))
        return prog

    def test_many_random_programs_match_reference(self):
        rng = random.Random(20260922)
        mem = words_to_bytes([rng.randrange(0, 1 << 32)
                              for _ in range(8)])
        matched = 0
        faults = 0
        halts = 0
        for trial in range(300):
            n = rng.randrange(1, 24)
            prog = self._gen_program(rng, n)
            p = run_pipeline(prog, mem)
            s = run_sequential(prog, mem)
            self.assertEqual(
                p.halted, s.halted,
                f"trial {trial} halted mismatch: {[str(x) for x in prog]}")
            self.assertEqual(
                None if p.fault is None else p.fault.kind,
                None if s.fault is None else s.fault.kind,
                f"trial {trial} fault kind mismatch")
            self.assertEqual(
                None if p.fault is None else p.fault.address,
                None if s.fault is None else s.fault.address,
                f"trial {trial} fault addr mismatch")
            self.assertEqual(
                p.final_registers, s.final_registers, f"trial {trial} regs")
            self.assertEqual(
                p.final_memory, s.final_memory, f"trial {trial} memory")
            matched += 1
            faults += p.fault is not None
            halts += p.halted
        # 300 例中应同时覆盖停机与异常两类结局，避免空泛比较。
        self.assertGreater(halts, 0)
        self.assertGreater(faults, 0)


class BoundsTests(unittest.TestCase):
    def test_max_instructions_program_runs(self):
        # 200 条：199 条 ADDI 链 + HALT，周期数远小于 2000。
        prog = [Instruction(Opcode.ADDI, a=1, b=0, imm=1)]
        for _ in range(198):
            prog.append(Instruction(Opcode.ADDI, a=1, b=1, imm=1))
        prog.append(Instruction(Opcode.HALT))
        self.assertEqual(len(prog), 200)
        p = run_pipeline(prog, bytes(16))
        s = run_sequential(prog, bytes(16))
        self.assertTrue(p.halted)
        self.assertEqual(p.final_registers[1], s.final_registers[1])
        self.assertEqual(p.final_registers[1], 199)
        self.assertLessEqual(p.cycles, 2000)

    def test_cycle_limit_hit_on_long_loop(self):
        # 无限循环（自跳 BEQ）在小预算下报预算耗尽，且区别于故障/停机。
        prog = [Instruction(Opcode.BEQ, b=0, c=0, imm=0)]  # pc0 自跳
        from pipeline_cpu import CycleLimitExceeded
        with self.assertRaises(CycleLimitExceeded):
            run_pipeline(prog, bytes(16), max_cycles=50)


if __name__ == "__main__":
    unittest.main()
