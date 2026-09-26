"""MMIO 设备访问：恰好一次、只在提交点、与顺序参考逐条对照。

覆盖：分支冲刷不碰设备、设备空读精确异常、同地址连续操作、
HALT 前最后一次写、异常后年轻指令不碰设备、MMIO 与普通 store 的顺序、
提交日志中的实际访问点。设备为本地状态机，不连接任何外设。
"""

import random
import unittest

from tinycpu import (
    Fault,
    Instruction,
    MMIOBus,
    MMIOEmptyError,
    memory_from_words,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

READ_ADDR = 0x00008000
WRITE_ADDR = 0x00008004
MEM = memory_from_words([0] * 8)


def fresh_bus(values=()):
    return MMIOBus(READ_ADDR, WRITE_ADDR, list(values))


def run_both(prog, values=(), mem=MEM):
    """同一程序分别跑流水线与顺序机（各自一条全新总线），返回四元组。"""
    bus_p, bus_s = fresh_bus(values), fresh_bus(values)
    res = run(prog, mem, mmio=bus_p)
    seq = run_sequential(prog, mem, mmio=bus_s)
    return res, seq, bus_p, bus_s


def assert_same(testcase, res, seq, bus_p, bus_s):
    testcase.assertEqual(res.regs, seq.regs)
    testcase.assertEqual(res.memory, seq.memory)
    testcase.assertEqual(res.halted, seq.halted)
    testcase.assertEqual(res.fault, seq.fault)
    testcase.assertEqual(res.fault_pc, seq.fault_pc)
    # 逐条对照退休轨迹的架构效果（含 MMIO 访问标记）。
    testcase.assertEqual(len(res.trace), len(seq.trace))
    for rp, rs in zip(res.trace, seq.trace):
        testcase.assertEqual(
            (rp.pc, rp.op, rp.fault, rp.rd, rp.value, rp.addr, rp.mem_value, rp.mmio),
            (rs.pc, rs.op, rs.fault, rs.rd, rs.value, rs.addr, rs.mem_value, rs.mmio),
        )
    # 设备队列、读日志、写记录完全一致。
    testcase.assertEqual(bus_p.snapshot(), bus_s.snapshot())


class TestMMIOBasic(unittest.TestCase):
    def test_read_pops_and_writes_back(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.load(2, 1, 0),
            Instruction.load(3, 1, 0),
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[0xDEADBEEF, 7])
        self.assertTrue(res.halted)
        self.assertEqual(res.regs[2], 0xDEADBEEF)
        self.assertEqual(res.regs[3], 7)
        self.assertEqual(bp.snapshot(), ((), (0xDEADBEEF, 7), ()))
        assert_same(self, res, seq, bp, bs)

    def test_write_appends_record(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.addi(4, 0, 42),
            Instruction.store(4, 1, 4),   # -> WRITE_ADDR
            Instruction.addi(4, 4, 1),
            Instruction.store(4, 1, 4),
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog)
        self.assertEqual(bp.snapshot()[2], (42, 43))
        self.assertEqual(res.memory, MEM)  # 普通内存未被 MMIO 触碰
        assert_same(self, res, seq, bp, bs)

    def test_consecutive_same_address_ops(self):
        """同地址连续读按提交顺序依次弹出；连续写按序追加。"""
        prog = [Instruction.addi(1, 0, READ_ADDR)]
        for i in range(3):
            prog.append(Instruction.load(2 + i, 1, 0))
        for i in range(3):
            prog.append(Instruction.store(2 + i, 1, 4))
        prog.append(Instruction.halt())
        res, seq, bp, bs = run_both(prog, values=[10, 20, 30])
        self.assertEqual(res.regs[2:5], (10, 20, 30))
        self.assertEqual(bp.snapshot(), ((), (10, 20, 30), (10, 20, 30)))
        assert_same(self, res, seq, bp, bs)

    def test_last_write_before_halt_commits_once(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.addi(4, 0, 0x1234),
            Instruction.store(4, 1, 4),   # HALT 前最后一次设备写
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog)
        self.assertTrue(res.halted)
        self.assertEqual(bp.snapshot()[2], (0x1234,))  # 恰好一次
        # 提交日志：写在 HALT 之前退休，带 mmio 标记与实际访问周期。
        mmio_recs = [r for r in res.trace if r.mmio == "write"]
        self.assertEqual(len(mmio_recs), 1)
        self.assertEqual(mmio_recs[0].mem_value, 0x1234)
        self.assertLess(mmio_recs[0].cycle, res.trace[-1].cycle)
        assert_same(self, res, seq, bp, bs)


class TestMMIOCommitOnly(unittest.TestCase):
    def test_flushed_branch_path_never_touches_device(self):
        """错误路径上的 MMIO 读/写被冲刷，设备队列与写记录不变。"""
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.beq(0, 0, 3),      # 恒跳转，跳过下面两条
            Instruction.load(2, 1, 0),     # 错误路径：设备读
            Instruction.store(2, 1, 4),    # 错误路径：设备写
            Instruction.addi(5, 0, 9),
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[77])
        self.assertTrue(res.halted)
        self.assertEqual(res.regs[5], 9)
        self.assertEqual(bp.snapshot(), ((77,), (), ()))  # 设备完全没被碰
        self.assertFalse(any(r.mmio for r in res.trace))
        assert_same(self, res, seq, bp, bs)

    def test_younger_mmio_store_after_fault_not_executed(self):
        """非法访存之后的年轻 MMIO 写不得发生；老指令正常提交。"""
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.addi(4, 0, 55),
            Instruction.store(4, 0, 3),    # 非对齐 -> Fault.ADDR（在 WB 报告）
            Instruction.store(4, 1, 4),    # 年轻 MMIO 写：必须被丢弃
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog)
        self.assertEqual(res.fault, Fault.ADDR)
        self.assertEqual(res.fault_pc, 2)
        self.assertEqual(bp.snapshot()[2], ())  # 设备写记录为空
        assert_same(self, res, seq, bp, bs)

    def test_mmio_order_vs_normal_store(self):
        """MMIO 写与普通内存 store 按程序序在 WB 依次提交。"""
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.addi(4, 0, 1),
            Instruction.store(4, 0, 0),    # 普通内存写 mem[0]
            Instruction.addi(4, 0, 2),
            Instruction.store(4, 1, 4),    # MMIO 写
            Instruction.addi(4, 0, 3),
            Instruction.store(4, 0, 4),    # 普通内存写 mem[4]
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog)
        self.assertEqual(res.memory[:8], bytes(memory_from_words([1, 3])[:8]))
        self.assertEqual(bp.snapshot()[2], (2,))
        # 提交日志中的访问点顺序：mem[0] -> MMIO -> mem[4]
        accesses = [
            (r.cycle, r.addr, r.mmio) for r in res.trace if r.op.value == "STORE"
        ]
        self.assertEqual([a[1] for a in accesses], [0, WRITE_ADDR, 4])
        self.assertEqual([a[2] for a in accesses], [None, "write", None])
        self.assertEqual(
            [a[0] for a in accesses], sorted(a[0] for a in accesses)
        )
        assert_same(self, res, seq, bp, bs)


class TestMMIOEmptyRead(unittest.TestCase):
    def test_empty_read_is_precise_fault(self):
        """空队列读报 MMIO_EMPTY：老指令已提交，年轻指令（含 MMIO 写）不提交。"""
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.addi(5, 0, 8),
            Instruction.store(5, 0, 0),    # 老指令：普通写必须落内存
            Instruction.load(2, 1, 0),     # 空读 -> Fault.MMIO_EMPTY
            Instruction.store(5, 1, 4),    # 年轻 MMIO 写：不得发生
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[])
        self.assertEqual(res.fault, Fault.MMIO_EMPTY)
        self.assertEqual(res.fault_pc, 3)
        self.assertEqual(res.memory[:4], (8).to_bytes(4, "little"))
        self.assertEqual(bp.snapshot(), ((), (), ()))
        # 故障记录在提交日志中，fault 周期即访问点。
        self.assertEqual(res.trace[-1].fault, Fault.MMIO_EMPTY)
        self.assertEqual(res.trace[-1].cycle, res.fault_cycle)
        assert_same(self, res, seq, bp, bs)

    def test_queue_drains_exactly_once_per_retired_load(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.load(2, 1, 0),
            Instruction.load(3, 1, 0),     # 空读 -> 故障
            Instruction.load(4, 1, 0),     # 年轻：不得弹队列
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[111])
        self.assertEqual(res.fault, Fault.MMIO_EMPTY)
        self.assertEqual(res.regs[2], 111)
        self.assertEqual(bp.snapshot(), ((), (111,), ()))
        assert_same(self, res, seq, bp, bs)


class TestMMIOValidation(unittest.TestCase):
    def test_direction_mismatch_is_addr_fault(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.load(2, 1, 4),     # LOAD 命中写寄存器 -> ADDR
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[1])
        self.assertEqual(res.fault, Fault.ADDR)
        self.assertEqual(bp.snapshot(), ((1,), (), ()))
        assert_same(self, res, seq, bp, bs)

    def test_bus_constructor_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            MMIOBus(READ_ADDR, READ_ADDR)
        with self.assertRaises(ValueError):
            MMIOBus(2, WRITE_ADDR)          # 未对齐
        with self.assertRaises(TypeError):
            MMIOBus(READ_ADDR, WRITE_ADDR, [True])
        with self.assertRaises(ValueError):
            MMIOBus(READ_ADDR, WRITE_ADDR, [1 << 32])
        bus = fresh_bus()
        with self.assertRaises(MMIOEmptyError):
            bus.read()

    def test_mmio_param_type_checked(self):
        prog = [Instruction.halt()]
        with self.assertRaises(TypeError):
            run(prog, MEM, mmio=object())
        with self.assertRaises(TypeError):
            run_sequential(prog, MEM, mmio="bus")

    def test_no_mmio_keeps_baseline_behavior(self):
        """不传 mmio 时，对 MMIO 地址的访问仍是普通 ADDR 故障。"""
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.load(2, 1, 0),
            Instruction.halt(),
        ]
        res = run(prog, MEM)
        self.assertEqual(res.fault, Fault.ADDR)


class TestMMIOCommitLog(unittest.TestCase):
    def test_commit_log_marks_actual_access_points(self):
        prog = [
            Instruction.addi(1, 0, READ_ADDR),
            Instruction.load(2, 1, 0),
            Instruction.store(2, 1, 4),
            Instruction.halt(),
        ]
        res, seq, bp, bs = run_both(prog, values=[0xAB])
        reads = [r for r in res.trace if r.mmio == "read"]
        writes = [r for r in res.trace if r.mmio == "write"]
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(writes), 1)
        self.assertEqual(reads[0].value, 0xAB)
        self.assertEqual(writes[0].mem_value, 0xAB)
        # 访问点周期严格递增且都在退休轨迹上。
        self.assertLess(reads[0].cycle, writes[0].cycle)
        # 顺序机轨迹同样带 mmio 标记。
        self.assertEqual([e.mmio for e in seq.trace], [None, "read", "write", None])
        assert_same(self, res, seq, bp, bs)


class TestMMIOEquivalenceRandom(unittest.TestCase):
    """确定性随机程序（含 MMIO 访问）的流水线/顺序机差分测试。"""

    def test_random_programs(self):
        for seed in range(120):
            rng = random.Random(10_000 + seed)
            prog, init, values = self._make_program(rng)
            mem = memory_from_words([rng.randrange(0, 1 << 32) for _ in range(8)])
            bus_p = MMIOBus(READ_ADDR, WRITE_ADDR, list(values))
            bus_s = MMIOBus(READ_ADDR, WRITE_ADDR, list(values))
            res = run(prog, mem, init, mmio=bus_p)
            seq = run_sequential(prog, mem, init, mmio=bus_s)
            assert_same(self, res, seq, bus_p, bus_s)

    @staticmethod
    def _make_program(rng):
        from tinycpu import Op

        length = 16
        prog = []
        for pc in range(length - 1):
            op = rng.choices(
                [Op.ADD, Op.SUB, Op.ADDI, Op.LOAD, Op.STORE, Op.BEQ],
                weights=[3, 2, 4, 4, 4, 2], k=1,
            )[0]
            if op in (Op.ADD, Op.SUB):
                prog.append(Instruction(op, rd=rng.randrange(1, 32),
                                        rs1=rng.randrange(32), rs2=rng.randrange(32)))
            elif op == Op.ADDI:
                prog.append(Instruction.addi(rng.randrange(0, 32), rng.randrange(32),
                                             rng.randrange(-40, 41)))
            elif op in (Op.LOAD, Op.STORE):
                # 一半概率把基址设为 MMIO 读寄存器，偏移 0/4 命中两个设备。
                base = rng.randrange(32)
                off = rng.choice([0, 0, 4, -4, 8])
                if rng.random() < 0.5:
                    base = 31  # r31 固定为 READ_ADDR（见 init）
                    off = rng.choice([0, 4])
                if op == Op.LOAD:
                    prog.append(Instruction.load(rng.randrange(0, 32), base, off))
                else:
                    prog.append(Instruction.store(rng.randrange(32), base, off))
            else:
                max_off = length - 1 - pc
                prog.append(Instruction.beq(rng.randrange(32), rng.randrange(32),
                                            rng.randrange(1, max_off + 1)))
        prog.append(Instruction.halt())
        init = [0] * 32
        for i in range(1, 31):
            init[i] = rng.choice([0, 4, 8, rng.randrange(0, 1 << 20)])
        init[31] = READ_ADDR
        values = [rng.randrange(0, 1 << 32) for _ in range(rng.randrange(0, 6))]
        return prog, init, values


if __name__ == "__main__":
    unittest.main()
