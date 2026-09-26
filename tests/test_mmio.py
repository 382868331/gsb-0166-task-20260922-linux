"""MMIO 设备寄存器：只在提交点访问、恰好一次，与顺序参考逐条对照。

覆盖：分支冲刷不触设备、设备空读错误（Fault.DEV）、同地址连续操作、
HALT 前最后一次写、MMIO 与普通 store 的顺序、设备读值写回与前递。
"""

from __future__ import annotations

import unittest

from tinycpu import (
    Fault,
    Instruction,
    MMIO_READ_ADDR,
    MMIO_WRITE_ADDR,
    MMIODevices,
    memory_from_words,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

I = Instruction
MEM_WORDS = 8


def run_both(prog, read_values=(), mem_words=MEM_WORDS):
    """同一程序分别跑流水线与顺序参考（各自独立设备实例），返回四元组。"""
    mem = memory_from_words([0] * mem_words)
    dp = MMIODevices(read_values)
    ds = MMIODevices(read_values)
    pipe = run(prog, bytearray(mem), devices=dp)
    seq = run_sequential(prog, bytearray(mem), devices=ds)
    return pipe, seq, dp, ds


def assert_full_match(tc, pipe, seq, dp, ds):
    """逐条对照退休轨迹、寄存器、内存、设备队列与写记录。"""
    tc.assertEqual(pipe.regs, seq.regs)
    tc.assertEqual(pipe.memory, seq.memory)
    tc.assertEqual(dp.pending_reads, ds.pending_reads)
    tc.assertEqual(dp.write_log, ds.write_log)
    tc.assertEqual(pipe.fault, seq.fault)
    tc.assertEqual(pipe.fault_pc, seq.fault_pc)
    tc.assertEqual(len(pipe.trace), len(seq.trace))
    for pt, st in zip(pipe.trace, seq.trace):
        tc.assertEqual(
            (pt.pc, pt.op, pt.fault, pt.rd, pt.value, pt.addr, pt.mem_value, pt.mmio),
            (st.pc, st.op, st.fault, st.rd, st.value, st.addr, st.mem_value, st.mmio),
        )


class TestMmioBasic(unittest.TestCase):
    def test_read_pops_and_writes_back(self):
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.load(2, 1, 0),          # 弹出 0xAB
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[0xAB, 0xCD])
        self.assertEqual(pipe.regs[2], 0xAB)
        self.assertEqual(dp.pending_reads, (0xCD,))   # 恰好弹一次
        rec = pipe.trace[1]
        self.assertEqual((rec.mmio, rec.rd, rec.value, rec.addr),
                         ("read", 2, 0xAB, MMIO_READ_ADDR))
        assert_full_match(self, pipe, seq, dp, ds)

    def test_write_appends_record(self):
        prog = [
            I.addi(1, 0, MMIO_WRITE_ADDR),
            I.addi(2, 0, 77),
            I.store(2, 1, 0),
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog)
        self.assertEqual(dp.write_log, (77,))
        rec = pipe.trace[2]
        self.assertEqual((rec.mmio, rec.addr, rec.mem_value),
                         ("write", MMIO_WRITE_ADDR, 77))
        assert_full_match(self, pipe, seq, dp, ds)

    def test_consecutive_same_address_ops(self):
        """同地址连续读两次弹两次、写两次记两条，顺序与值都正确。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.addi(2, 0, MMIO_WRITE_ADDR),
            I.load(3, 1, 0),          # 弹 10
            I.load(4, 1, 0),          # 紧接同地址再弹 20
            I.store(3, 2, 0),         # 同地址连续写
            I.store(4, 2, 0),
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[10, 20, 30])
        self.assertEqual((pipe.regs[3], pipe.regs[4]), (10, 20))
        self.assertEqual(dp.pending_reads, (30,))
        self.assertEqual(dp.write_log, (10, 20))
        mmio_recs = [t for t in pipe.trace if t.mmio]
        self.assertEqual([t.mmio for t in mmio_recs],
                         ["read", "read", "write", "write"])
        assert_full_match(self, pipe, seq, dp, ds)

    def test_last_write_before_halt(self):
        """HALT 前最后一条设备写必须真实提交（HALT 不抢在它前面）。"""
        prog = [
            I.addi(1, 0, MMIO_WRITE_ADDR),
            I.addi(2, 0, 0xFF),
            I.store(2, 1, 0),         # HALT 前最后一次写
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog)
        self.assertTrue(pipe.halted)
        self.assertEqual(dp.write_log, (0xFF,))
        # 提交日志里设备写出现在 HALT 之前。
        kinds = [(t.op.name, t.mmio) for t in pipe.trace]
        self.assertEqual(kinds[-2:], [("STORE", "write"), ("HALT", None)])
        assert_full_match(self, pipe, seq, dp, ds)

    def test_mmio_read_value_forwarded(self):
        """设备读值经 load-use 停顿与前递进入后续运算，不预测不校正。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.load(2, 1, 0),          # 弹 41
            I.addi(3, 2, 1),          # load-use：42
            I.add(4, 2, 3),           # 前递链：41+42=83
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[41])
        self.assertEqual((pipe.regs[3], pipe.regs[4]), (42, 83))
        assert_full_match(self, pipe, seq, dp, ds)


class TestMmioSpeculationSafety(unittest.TestCase):
    def test_flushed_path_never_touches_device(self):
        """错误分支上的设备读/写被冲刷，设备队列与写记录零变化。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.addi(2, 0, MMIO_WRITE_ADDR),
            I.addi(3, 0, 5),
            I.beq(3, 3, 3),           # 恒真 -> pc6，冲刷 pc4/pc5
            I.load(4, 1, 0),          # 被冲刷：不得弹队列
            I.store(3, 2, 0),         # 被冲刷：不得追加记录
            I.addi(5, 0, 9),          # pc6 分支目标
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[123])
        self.assertEqual(dp.pending_reads, (123,))    # 队列未动
        self.assertEqual(dp.write_log, ())            # 无设备写
        self.assertEqual([t.pc for t in pipe.trace], [0, 1, 2, 3, 6, 7])
        self.assertFalse(any(t.mmio for t in pipe.trace))
        assert_full_match(self, pipe, seq, dp, ds)

    def test_younger_after_fault_never_touches_device(self):
        """非法访存之后的年轻设备操作一律不发生。"""
        prog = [
            I.addi(1, 0, 5),          # 非对齐地址
            I.addi(2, 0, MMIO_WRITE_ADDR),
            I.addi(3, 0, 55),
            I.store(3, 1, 0),         # pc3：非法地址 -> WB 报 ADDR
            I.store(3, 2, 0),         # pc4：年轻设备写，禁止发生
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog)
        self.assertEqual(pipe.fault, Fault.ADDR)
        self.assertEqual(pipe.fault_pc, 3)
        self.assertEqual(dp.write_log, ())
        assert_full_match(self, pipe, seq, dp, ds)


class TestMmioEmptyRead(unittest.TestCase):
    def test_empty_read_fault_dev(self):
        """设备空读：Fault.DEV 在 WB 报告，该读不提交，年轻写不发生。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.addi(2, 0, MMIO_WRITE_ADDR),
            I.load(3, 1, 0),          # 空队列 -> DEV
            I.store(3, 2, 0),         # 年轻设备写，禁止发生
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[])
        self.assertEqual(pipe.fault, Fault.DEV)
        self.assertEqual(pipe.fault_pc, 2)
        self.assertIsNotNone(pipe.fault_cycle)
        self.assertEqual(dp.pending_reads, ())
        self.assertEqual(dp.write_log, ())
        self.assertEqual(pipe.regs[3], 0)             # 故障读不写回
        fault_rec = pipe.trace[-1]
        self.assertEqual((fault_rec.fault, fault_rec.pc), (Fault.DEV, 2))
        assert_full_match(self, pipe, seq, dp, ds)

    def test_empty_read_after_consumed_queue(self):
        """队列被弹空后再读才出错：第一次读正常提交。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.load(2, 1, 0),          # 弹走唯一元素
            I.load(3, 1, 0),          # 空 -> DEV
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[7])
        self.assertEqual(pipe.regs[2], 7)
        self.assertEqual(pipe.fault, Fault.DEV)
        self.assertEqual(pipe.fault_pc, 2)
        assert_full_match(self, pipe, seq, dp, ds)


class TestMmioOrdering(unittest.TestCase):
    def test_mmio_and_normal_store_order(self):
        """MMIO 写与普通内存写交替，提交日志顺序与程序顺序一致。"""
        prog = [
            I.addi(1, 0, MMIO_WRITE_ADDR),
            I.addi(2, 0, 0),          # 内存基址 0
            I.addi(3, 0, 11),
            I.store(3, 2, 0),         # mem[0] = 11
            I.store(3, 1, 0),         # dev <- 11
            I.addi(3, 3, 1),          # 12
            I.store(3, 2, 4),         # mem[4] = 12
            I.store(3, 1, 0),         # dev <- 12
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog)
        self.assertEqual(dp.write_log, (11, 12))
        self.assertEqual(
            [t.mem_value for t in pipe.trace if t.op.name == "STORE"],
            [11, 11, 12, 12],
        )
        self.assertEqual(
            [t.mmio for t in pipe.trace if t.op.name == "STORE"],
            [None, "write", None, "write"],
        )
        assert_full_match(self, pipe, seq, dp, ds)

    def test_mmio_addr_without_devices_is_addr_fault(self):
        """不传 devices 时 MMIO 地址按非法访存处理（保持基线行为）。"""
        prog = [
            I.addi(1, 0, MMIO_WRITE_ADDR),
            I.addi(2, 0, 1),
            I.store(2, 1, 0),
            I.halt(),
        ]
        res = run(prog, memory_from_words([0] * 4))
        self.assertEqual(res.fault, Fault.ADDR)
        self.assertEqual(res.fault_pc, 2)

    def test_devices_type_checked(self):
        prog = [I.halt()]
        with self.assertRaises(TypeError):
            run(prog, memory_from_words([0]), devices=object())
        with self.assertRaises(TypeError):
            run_sequential(prog, memory_from_words([0]), devices="dev")


class TestMmioDeviceModel(unittest.TestCase):
    def test_read_values_validated(self):
        with self.assertRaises(ValueError):
            MMIODevices([1 << 32])
        with self.assertRaises(TypeError):
            MMIODevices([True])
        with self.assertRaises(TypeError):
            MMIODevices([1.5])

    def test_read_to_r0_still_pops(self):
        """目标为 r0 的设备读：寄存器写被丢弃，但弹队列动作仍提交一次。"""
        prog = [
            I.addi(1, 0, MMIO_READ_ADDR),
            I.load(0, 1, 0),
            I.halt(),
        ]
        pipe, seq, dp, ds = run_both(prog, read_values=[9, 8])
        self.assertEqual(dp.pending_reads, (8,))
        self.assertEqual(pipe.regs[0], 0)
        assert_full_match(self, pipe, seq, dp, ds)


if __name__ == "__main__":
    unittest.main()

