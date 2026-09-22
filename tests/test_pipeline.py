"""流水线结构行为测试：时序、前递、load-use 停顿时、分支冲刷。"""

import unittest

from pipeline_cpu import (
    Instruction, Opcode, run_pipeline,
)


def W(*xs: int) -> bytes:
    return b"".join((x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)


class BasicTimingTests(unittest.TestCase):
    def test_single_halt_takes_five_cycles(self):
        r = run_pipeline([Instruction(Opcode.HALT)], bytes(16))
        self.assertTrue(r.halted)
        self.assertEqual(r.cycles, 5)
        # 每周期恰好一个槽占据，沿 IF->WB 流动。
        for i, t in enumerate(r.trace):
            slots = [t.if_stage.seq, t.id_stage.seq, t.ex_stage.seq,
                     t.mem_stage.seq, t.wb_stage.seq]
            self.assertEqual(slots[i], 0)
            self.assertEqual(sum(s is not None for s in slots), 1)
        self.assertEqual(r.retirements[0].cycle, 5)
        self.assertTrue(r.retirements[0].halted)

    def test_steady_throughput_one_retire_per_cycle(self):
        prog = [Instruction(Opcode.ADDI, a=i % 30 + 1, b=0, imm=i)
                for i in range(10)] + [Instruction(Opcode.HALT)]
        r = run_pipeline(prog, bytes(16))
        # 填满后每周期退休一条；只允许末周期出现 halt 注记。
        self.assertEqual(r.cycles, len(prog) + 4)
        self.assertEqual(
            [x.seq for x in r.retirements], list(range(len(prog))))
        self.assertTrue(
            all(t.note in ("", "halt") for t in r.trace))

    def test_r0_is_hardwired_zero(self):
        prog = [
            Instruction(Opcode.ADDI, a=0, b=0, imm=123),
            Instruction(Opcode.ADD, a=1, b=0, c=0),
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.final_registers[0], 0)
        self.assertEqual(r.final_registers[1], 0)
        wb0 = next(x for x in r.retirements if x.seq == 0)
        self.assertIsNone(wb0.value)  # 写 r0 被丢弃


class ForwardingTests(unittest.TestCase):
    def test_exmem_forward_arithmetic_chain(self):
        # r1=5; r2=r1+1; r3=r2+r1 —— 无停顿完成，结果正确即证明前递。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=5),
            Instruction(Opcode.ADDI, a=2, b=1, imm=1),
            Instruction(Opcode.ADD, a=3, b=2, c=1),
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.final_registers[1:4], (5, 6, 11))
        self.assertTrue(all("stall" not in t.note for t in r.trace))

    def test_newest_value_wins_consecutive_same_register(self):
        # 连续写 r1；依赖指令必须拿到最新值（EX/MEM 优先于 MEM/WB）。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=10),
            Instruction(Opcode.ADDI, a=1, b=0, imm=20),
            Instruction(Opcode.ADDI, a=1, b=0, imm=30),
            Instruction(Opcode.ADDI, a=2, b=1, imm=0),  # 紧邻依赖，前递
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.final_registers[1], 30)
        self.assertEqual(r.final_registers[2], 30)
        self.assertTrue(all("stall" not in t.note for t in r.trace))

    def test_sub_and_wrap_around(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=0),
            Instruction(Opcode.ADDI, a=2, b=0, imm=1),
            Instruction(Opcode.SUB, a=3, b=1, c=2),   # 0-1 = 0xFFFFFFFF
            Instruction(Opcode.ADD, a=4, b=3, c=2),   # FFFFFFFF+1 = 0
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.final_registers[3], 0xFFFFFFFF)
        self.assertEqual(r.final_registers[4], 0)

    def test_forward_around_stall(self):
        # LOAD r1 后隔一条再用 r1：MEM/WB 前递，无停顿。
        prog = [
            Instruction(Opcode.LOAD, a=1, b=0, imm=0),
            Instruction(Opcode.ADDI, a=2, b=0, imm=7),
            Instruction(Opcode.ADD, a=3, b=1, c=2),  # MEM/WB 前递
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, W(41))
        self.assertEqual(r.final_registers[3], 48)
        self.assertTrue(all("stall" not in t.note for t in r.trace))


class LoadUseTests(unittest.TestCase):
    def test_load_use_stalls_exactly_one_cycle(self):
        prog = [
            Instruction(Opcode.LOAD, a=1, b=0, imm=0),
            Instruction(Opcode.ADDI, a=2, b=1, imm=5),  # 紧邻使用
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, W(100))
        stall_cycles = [t.cycle for t in r.trace if "stall" in t.note]
        self.assertEqual(stall_cycles, [3])  # LOAD(seq0)在 EX，恰好停一周期
        self.assertEqual(r.final_registers[2], 105)
        # 停顿时 ID/IF 冻结同一指令序号，EX 仍是该 LOAD。
        t3 = r.trace[2]
        self.assertEqual(t3.ex_stage.seq, 0)
        self.assertEqual(t3.id_stage.seq, 1)
        self.assertTrue(t3.id_stage.stalled)
        self.assertEqual(t3.if_stage.seq, 2)
        self.assertTrue(t3.if_stage.stalled)
        # 停顿只增加一个周期。
        self.assertEqual(r.cycles, len(prog) + 4 + 1)

    def test_load_use_into_store_base_and_data(self):
        prog = [
            Instruction(Opcode.LOAD, a=1, b=0, imm=0),   # r1 = mem[0]=8
            Instruction(Opcode.STORE, b=1, c=1, imm=0),  # 紧邻：base+data
            Instruction(Opcode.HALT),
        ]
        mem = W(8, 0)
        r = run_pipeline(prog, mem)
        stall = [t for t in r.trace if "stall" in t.note]
        self.assertEqual(stall[0].cycle, 3)
        self.assertEqual(r.final_memory, W(8, 0))

    def test_load_use_through_beq(self):
        # LOAD r1=99 后紧邻 BEQ r1,r0（不等不跳）；停顿一次且比较正确。
        prog = [
            Instruction(Opcode.LOAD, a=1, b=0, imm=0),
            Instruction(Opcode.BEQ, b=1, c=0, imm=8),
            Instruction(Opcode.ADDI, a=2, b=0, imm=1),
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, W(99) + bytes(12))
        self.assertEqual(len([t for t in r.trace if "stall" in t.note]), 1)
        self.assertEqual(r.final_registers[2], 1)  # 未跳转，顺序执行

    def test_load_to_r0_does_not_stall(self):
        prog = [
            Instruction(Opcode.LOAD, a=0, b=0, imm=0),
            Instruction(Opcode.ADDI, a=1, b=0, imm=3),
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, W(9))
        self.assertTrue(all("stall" not in t.note for t in r.trace))


class BranchTests(unittest.TestCase):
    def test_branch_taken_flushes_two_young_instructions(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=1),       # 0
            Instruction(Opcode.BEQ, b=1, c=1, imm=8),       # 1 -> seq3
            Instruction(Opcode.ADDI, a=2, b=0, imm=77),     # 2 被冲刷
            Instruction(Opcode.HALT),                        # 3
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertTrue(r.halted)
        self.assertEqual(r.final_registers[2], 0)
        flush_t = next(t for t in r.trace if "flush" in t.note)
        self.assertEqual(flush_t.ex_stage.seq, 1)   # BEQ 在 EX
        self.assertTrue(flush_t.id_stage.flushed)
        self.assertEqual(flush_t.id_stage.seq, 2)   # 被冲刷的年轻指令
        # 退休轨迹中 seq2 从未出现。
        self.assertEqual([x.seq for x in r.retirements], [0, 1, 3])

    def test_branch_not_taken_no_flush(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=1),
            Instruction(Opcode.BEQ, b=1, c=0, imm=8),  # 不等
            Instruction(Opcode.ADDI, a=2, b=0, imm=5),
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertTrue(all("flush" not in t.note for t in r.trace))
        self.assertEqual(r.final_registers[2], 5)

    def test_counted_loop(self):
        # r1=0; loop@seq1: r1+=1; r3=3; 相等则跳出到 seq5 HALT(pc12+8=20)，
        # 否则 seq4 无条件回跳 seq1(pc16-12=4)。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=0),         # 0
            Instruction(Opcode.ADDI, a=1, b=1, imm=1),         # 1
            Instruction(Opcode.ADDI, a=3, b=0, imm=3),         # 2
            Instruction(Opcode.BEQ, b=1, c=3, imm=8),          # 3 -> seq5 HALT
            Instruction(Opcode.BEQ, b=0, c=0, imm=-12),        # 4 -> seq1
            Instruction(Opcode.HALT),                          # 5
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertTrue(r.halted)
        self.assertEqual(r.final_registers[1], 3)
        flushes = [t for t in r.trace if "flush" in t.note]
        # 回跳冲刷 2 次 + 跳出冲刷 1 次 = 3 次。
        self.assertEqual(len(flushes), 3)


class PreciseFaultTests(unittest.TestCase):
    def test_illegal_store_reported_at_wb_without_write(self):
        # r1=201(奇地址); 老 STORE mem[0]=77 正常提交；
        # 故障 STORE mem[r1]=8 在 WB 报告；其后的 HALT 更年轻，不退休。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=201),       # 0
            Instruction(Opcode.ADDI, a=2, b=0, imm=77),        # 1
            Instruction(Opcode.STORE, b=0, c=2, imm=0),        # 2 合法
            Instruction(Opcode.STORE, b=1, c=2, imm=0),        # 3 非法
            Instruction(Opcode.HALT),                          # 4 不退休
        ]
        mem = bytes(16)
        r = run_pipeline(prog, mem)
        self.assertFalse(r.halted)
        self.assertIsNotNone(r.fault)
        self.assertEqual(r.fault.kind, "bad_address")
        self.assertEqual(r.fault.address, 201)
        self.assertEqual(r.fault_seq, 3)
        # 异常前内存：seq2 的写入在，seq3 的写入不在。
        self.assertEqual(r.final_memory, (77).to_bytes(4, "little") + bytes(12))
        self.assertEqual([x.seq for x in r.retirements], [0, 1, 2, 3])
        self.assertEqual(r.retirements[-1].fault, r.fault)

    def test_illegal_load_reported_at_wb_no_writeback(self):
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=100),       # 0
            Instruction(Opcode.LOAD, a=2, b=1, imm=0),         # 1 越界
            Instruction(Opcode.HALT),                          # 2
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.fault.kind, "bad_address")
        self.assertEqual(r.fault.address, 100)
        self.assertEqual(r.final_registers[2], 0)  # 未写回

    def test_younger_store_after_fault_never_commits(self):
        # 故障 LOAD(seq1) 之后紧跟 STORE(seq2)：STORE 在故障退休同周期
        # 仅位于 MEM，必须被冻结丢弃，内存不变。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=999),       # 0
            Instruction(Opcode.LOAD, a=2, b=1, imm=0),         # 1 故障
            Instruction(Opcode.STORE, b=0, c=1, imm=0),        # 2 年轻store
            Instruction(Opcode.HALT),                          # 3
        ]
        mem = bytes(16)
        r = run_pipeline(prog, mem)
        self.assertEqual(r.fault_seq, 1)
        self.assertEqual(r.final_memory, bytes(16))
        self.assertEqual([x.seq for x in r.retirements], [0, 1])
        # 故障退休周期 seq2 正占据 MEM 槽且被冻结。
        t = r.trace[r.fault_cycle - 1]
        self.assertEqual(t.mem_stage.seq, 2)
        self.assertEqual(t.note, "fault")

    def test_fault_fetch_out_of_program(self):
        # 无条件跳转到程序之外（BEQ 是 seq0，pc=0）：bad_fetch 到 WB 退休。
        prog = [
            Instruction(Opcode.BEQ, b=0, c=0, imm=40),  # 0+40=40 越界
            Instruction(Opcode.HALT),
        ]
        r = run_pipeline(prog, bytes(16))
        self.assertIsNotNone(r.fault)
        self.assertEqual(r.fault.kind, "bad_fetch")
        self.assertEqual(r.fault.address, 40)

    def test_fall_through_end_bad_fetch(self):
        prog = [Instruction(Opcode.ADDI, a=1, b=0, imm=1)]  # 无 HALT
        r = run_pipeline(prog, bytes(16))
        self.assertEqual(r.fault.kind, "bad_fetch")
        self.assertEqual(r.fault.address, 4)
        self.assertEqual(r.final_registers[1], 1)  # 老指令正常提交

    def test_young_illegal_store_cancelled_by_branch(self):
        # 分支在 EX 冲刷其后非法 STORE：内存必须完好，程序正常 HALT。
        prog = [
            Instruction(Opcode.ADDI, a=1, b=0, imm=201),       # 0
            Instruction(Opcode.BEQ, b=0, c=0, imm=8),          # 1 -> seq3
            Instruction(Opcode.STORE, b=1, c=1, imm=0),        # 2 被冲刷
            Instruction(Opcode.HALT),                          # 3
        ]
        r = run_pipeline(prog, bytes(8))
        self.assertTrue(r.halted)
        self.assertIsNone(r.fault)
        self.assertEqual(r.final_memory, bytes(8))
        self.assertEqual([x.seq for x in r.retirements], [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
