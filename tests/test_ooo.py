"""4 项 ROB + 寄存器重命名乱序核心（tinycpu.ooo）的测试。

核对手段（期望值一律不取自被测实现）：
* 手算程序：最终寄存器/内存、退休序列、每个周期的派发/执行/完成/退休/
  回收事件都给出手工推导的常量；
* 结构断言：ROB 占用 ≤4、ALU 同周期执行即完成、LOAD 完成晚于执行、
  误预测回收含"已完成的年轻算术结果"、访存保序门控；
* 小规模穷举：在极小指令字母表上穷举 3~4 条指令的全部组合；
* 差分：独立顺序解释器 run_sequential（不导入 ooo/pipeline）给期望值，
  含数百个带长延迟 LOAD 与分支的确定性随机程序。
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from tinycpu import (
    BEQ,
    HALT,
    LOAD,
    STORE,
    Fault,
    Instruction,
    ProgramError,
    ROB_SIZE,
    LOAD_LATENCY,
    memory_from_words,
    words_from_memory,
)
from tinycpu.ooo import run_ooo
from tinycpu.pipeline import CycleLimitError
from tinycpu.sequential import run_sequential


MEM4 = memory_from_words([0, 0, 0, 0])


def _exec_cycle(result, pc, occurrence=0):
    seen = 0
    for c in result.cycles:
        if pc in c.execute:
            if seen == occurrence:
                return c.cycle
            seen += 1
    raise AssertionError(f"pc{pc} 从未执行")


def _complete_cycle(result, pc, occurrence=0):
    seen = 0
    for c in result.cycles:
        if pc in c.complete:
            if seen == occurrence:
                return c.cycle
            seen += 1
    raise AssertionError(f"pc{pc} 从未完成")


def _retire_cycle(result, pc, occurrence=0):
    seen = 0
    for c in result.cycles:
        if c.retire == pc:
            if seen == occurrence:
                return c.cycle
            seen += 1
    raise AssertionError(f"pc{pc} 从未退休")


class TestHandComputedBasics(unittest.TestCase):
    def test_basic_arith_final_values(self):
        # 手算：r1=5, r2=7, r3=5+7=12, r4=12-1=11
        prog = [
            Instruction.addi(1, 0, 5),
            Instruction.addi(2, 0, 7),
            Instruction.add(3, 1, 2),
            Instruction.addi(4, 3, -1),
            Instruction.halt(),
        ]
        r = run_ooo(prog, MEM4)
        self.assertTrue(r.halted)
        self.assertEqual((r.regs[1], r.regs[2], r.regs[3], r.regs[4]),
                         (5, 7, 12, 11))
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 4])
        self.assertEqual(r.regs[0], 0)

    def test_32bit_wraparound(self):
        prog = [
            Instruction.addi(1, 0, -1),          # 0xFFFFFFFF
            Instruction.addi(2, 1, 1),           # 0
            Instruction.sub(3, 0, 1),            # 1
            Instruction.addi(4, 0, 0x7FFFFFFF),
            Instruction.addi(4, 4, 1),           # 0x80000000 回绕
            Instruction.halt(),
        ]
        r = run_ooo(prog, MEM4)
        self.assertEqual((r.regs[1], r.regs[2], r.regs[3]),
                         (0xFFFFFFFF, 0, 1))
        self.assertEqual(r.regs[4], 0x80000000)

    def test_r0_writes_and_reads_are_zero(self):
        prog = [
            Instruction.addi(0, 0, 9),
            Instruction.add(1, 0, 0),
            Instruction.load(0, 0, 0),
            Instruction.addi(2, 1, 0),
            Instruction.halt(),
        ]
        r = run_ooo(prog, memory_from_words([123]))
        self.assertEqual(r.regs[0], 0)
        self.assertEqual(r.regs[1], 0)
        self.assertEqual(r.regs[2], 0)
        self.assertTrue(all(t.rd != 0 for t in r.trace))


class TestOutOfOrderCompletion(unittest.TestCase):
    def test_independent_alut_completes_before_older_load(self):
        # pc0 LOAD 需 LOAD_LATENCY 拍；pc1 独立 ADDI 应在 LOAD 完成前
        # 先乱序完成（整数算术乱序完成的直接时序证据）。
        prog = [
            Instruction.load(1, 0, 0),      # mem[0] = 77
            Instruction.addi(2, 0, 5),
            Instruction.add(3, 2, 0),
            Instruction.halt(),
        ]
        r = run_ooo(prog, memory_from_words([77]))
        self.assertEqual((r.regs[1], r.regs[2], r.regs[3]), (77, 5, 5))
        load_done = _complete_cycle(r, 0)
        alu_done = _complete_cycle(r, 1)
        self.assertLess(alu_done, load_done)
        # LOAD 从执行到完成至少跨 LOAD_LATENCY 拍。
        self.assertGreaterEqual(load_done - _exec_cycle(r, 0) + 1,
                                LOAD_LATENCY)
        # ALU/分支：执行与完成同周期。
        self.assertEqual(_exec_cycle(r, 1), alu_done)

    def test_waw_rename_takes_younger_value(self):
        # 同名 r1 连续写：消费者必须通过标签拿到最新（第二条）的值。
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(1, 0, 2),
            Instruction.add(2, 1, 1),
            Instruction.halt(),
        ]
        r = run_ooo(prog, MEM4)
        self.assertEqual(r.regs[1], 2)
        self.assertEqual(r.regs[2], 4)

    def test_dispatch_events_carry_rename_chain(self):
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(1, 0, 2),
            Instruction.add(3, 1, 1),
            Instruction.halt(),
        ]
        r = run_ooo(prog, MEM4)
        disps = [c.dispatch for c in r.cycles if c.dispatch is not None]
        self.assertEqual([d.pc for d in disps], [0, 1, 2, 3])
        self.assertEqual([d.seq for d in disps], [0, 1, 2, 3])
        # 第二次写 r1：old_tag 指向第一次写的标签 0；自己占标签 1。
        self.assertEqual(disps[1].rd, 1)
        self.assertEqual(disps[1].old_tag, 0)
        self.assertEqual(disps[1].seq, 1)
        # 纯消费者 ADD r3,r1,r1：重命名目标是 r3，两个源都指向当前属主 1。
        self.assertEqual(disps[2].rd, 3)
        self.assertEqual(disps[2].src, (1, 1))
        # HALT 不占重命名目标。
        self.assertIsNone(disps[3].rd)
        # 源操作数读架构寄存器时标签为 None。
        self.assertEqual(disps[0].src, (None, None))

    def test_rob_occupancy_bounded_and_full_stall(self):
        # 两条串行 LOAD（第二条被保序门控 + 数据依赖卡住）把门堵死，
        # 后端连续派发独立 ALU，必然出现 ROB(=4) 满停顿。
        prog = [
            Instruction.load(1, 0, 0),       # 0
            Instruction.load(2, 1, 0),       # 1 依赖 r1 且前面有更老 LOAD
            Instruction.addi(3, 0, 3),       # 2
            Instruction.addi(4, 0, 4),       # 3
            Instruction.addi(5, 0, 5),       # 4
            Instruction.addi(6, 0, 6),       # 5
            Instruction.halt(),              # 6
        ]
        r = run_ooo(prog, memory_from_words([0, 0, 0, 0]))
        for c in r.cycles:
            if c.dispatch is not None:
                self.assertLessEqual(c.dispatch.occupancy, ROB_SIZE)
                self.assertGreaterEqual(c.dispatch.occupancy, 1)
        self.assertTrue(
            any(c.dispatch_stall == "rob_full" for c in r.cycles),
            "ROB 满时必须出现 rob_full 前端停顿",
        )
        # 结果仍然正确（r1=r2=0，独立 ALU 全部生效）。
        self.assertTrue(r.halted and r.fault is None)
        self.assertEqual((r.regs[3], r.regs[4], r.regs[5], r.regs[6]),
                         (3, 4, 5, 6))


class TestBranchSquashRecovery(unittest.TestCase):
    def _squash_program(self):
        # 双 LOAD 串行把门堵 2 拍：pc3 独立 ALU 得以在老分支 pc2 解析前
        # 乱序完成；分支命中时 pc3 的"已完成年轻结果"必须被回收，
        # pc4 同周期派发也一并回收，RAT 恢复到分支前状态。
        return [
            Instruction.load(1, 0, 0),       # 0 r1=0
            Instruction.load(2, 1, 0),       # 1 r2=mem[0]=0
            Instruction.beq(2, 2, 4),        # 2 命中 -> pc6
            Instruction.addi(4, 0, 55),      # 3 先乱序完成，后被回收
            Instruction.addi(5, 0, 66),      # 4 派发即被回收
            Instruction.addi(6, 0, 77),      # 5 不执行
            Instruction.addi(7, 0, 7),       # 6 分支目标
            Instruction.halt(),              # 7
        ]

    def test_completed_young_result_is_reclaimed(self):
        prog = self._squash_program()
        r = run_ooo(prog, memory_from_words([0, 0, 0]))

        self.assertEqual(len(r.squashes), 1)
        sq = r.squashes[0]
        self.assertEqual((sq.branch_pc, sq.target), (2, 6))
        self.assertEqual(sq.squashed, (3, 4))
        # 关键：pc3 的 ALU 在回收之前已经完成（整数乱序完成后被丢弃）。
        self.assertEqual(sq.reclaimed_completed, (3,))

        # 退休序列跳过被回收的 pc3/pc4/pc5。
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 6, 7])
        # 被回收的写回绝不落架构寄存器。
        self.assertEqual((r.regs[4], r.regs[5], r.regs[6]), (0, 0, 0))
        self.assertEqual(r.regs[7], 7)
        self.assertTrue(r.halted and r.fault is None)

    def test_squash_cycle_event_consistency(self):
        r = run_ooo(self._squash_program(), memory_from_words([0, 0, 0]))
        sc = next(c for c in r.cycles if c.squash is not None)
        # 回收周期：分支当周期执行并完成；被回收者不出现在当周期退休。
        self.assertIn(2, sc.execute)
        self.assertIn(2, sc.complete)
        self.assertNotIn(3, [t.pc for t in r.trace])
        # 被回收 pc3 确实在更早的周期完成过。
        self.assertLess(_complete_cycle(r, 3), sc.cycle)
        # 回收之后前端从目标重新取指：目标 pc6 在更晚周期派发。
        d6 = next(c for c in r.cycles if c.dispatch and c.dispatch.pc == 6)
        self.assertGreater(d6.cycle, sc.cycle)

    def test_not_taken_branch_no_squash(self):
        prog = [
            Instruction.addi(1, 0, 3),
            Instruction.addi(2, 0, 4),
            Instruction.beq(1, 2, 2),     # 不命中
            Instruction.addi(3, 0, 9),
            Instruction.halt(),
        ]
        r = run_ooo(prog, MEM4)
        self.assertEqual(r.squashes, ())
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 4])
        self.assertEqual(r.regs[3], 9)

    def test_rat_restored_then_target_reads_correct_value(self):
        # 被回收路径写过 r1；恢复后目标指令读 r1 必须拿到分支前的老值。
        prog = [
            Instruction.addi(1, 0, 11),      # 0 r1=11（分支前老值）
            Instruction.addi(2, 0, 0),       # 1 r2=0
            Instruction.load(3, 0, 0),       # 2 堵门：mem 延迟
            Instruction.beq(0, 0, 3),        # 3 恒真 -> pc6
            Instruction.addi(1, 0, 99),      # 4 改写 r1，被回收
            Instruction.addi(4, 1, 0),       # 5 用错误 r1，也被回收
            Instruction.add(5, 1, 0),        # 6 目标：r5 必须 = 11
            Instruction.halt(),              # 7
        ]
        r = run_ooo(prog, memory_from_words([0]))
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 6, 7])
        self.assertEqual(r.regs[1], 11)
        self.assertEqual(r.regs[4], 0)
        self.assertEqual(r.regs[5], 11)


class TestMemoryOrdering(unittest.TestCase):
    def test_younger_load_executes_only_after_older_store_retires(self):
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.addi(2, 0, 0x41424344),
            Instruction.store(2, 1, 0),    # 2 老 STORE mem[0]
            Instruction.load(3, 1, 0),     # 3 年轻 LOAD 同一字
            Instruction.halt(),
        ]
        r = run_ooo(prog, memory_from_words([0]))
        # 年轻 LOAD 必须读到老 STOORE 的值（顺序访存），而非旧内存 0。
        self.assertEqual(r.regs[3], 0x41424344)
        self.assertEqual(words_from_memory(r.memory), [0x41424344])
        # 结构证据：LOAD 的执行周期晚于 STORE 的执行周期，
        # 且不早于 STORE 的退休周期（门控要求窗口内无更老访存）。
        self.assertGreater(_exec_cycle(r, 3), _exec_cycle(r, 2))
        self.assertGreaterEqual(_exec_cycle(r, 3), _retire_cycle(r, 2))

    def test_loads_issue_one_at_a_time_in_program_order(self):
        # 三条独立地址的 LOAD：执行（启动访存）顺序必须严格按程序序。
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.load(2, 1, 0),
            Instruction.load(3, 1, 4),
            Instruction.load(4, 1, 8),
            Instruction.halt(),
        ]
        r = run_ooo(prog, memory_from_words([10, 20, 30]))
        self.assertEqual((r.regs[2], r.regs[3], r.regs[4]), (10, 20, 30))
        self.assertLess(_exec_cycle(r, 1), _exec_cycle(r, 2))
        self.assertLess(_exec_cycle(r, 2), _exec_cycle(r, 3))

    def test_store_does_not_wait_for_younger_load_but_waits_older(self):
        # 老 STORE 不需要等年轻 ALU；年轻 LOAD 必须等老 STORE 退休。
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.addi(2, 0, 55),
            Instruction.store(2, 1, 0),
            Instruction.load(5, 1, 0),
            Instruction.add(6, 5, 2),       # 依赖装入
            Instruction.halt(),
        ]
        r = run_ooo(prog, memory_from_words([0]))
        seq = run_sequential(prog, memory_from_words([0]))
        self.assertEqual(r.memory, seq.memory)
        self.assertEqual(r.regs, seq.regs)
        self.assertEqual((r.regs[5], r.regs[6]), (55, 110))


class TestPreciseFaultAtRetire(unittest.TestCase):
    def test_fault_discards_young_completed_and_pending(self):
        # 老 STORE 合法提交；pc4 非法 LOAD 在 ROB 头部退休时才报告；
        # 年轻独立 ALU（可能已乱序完成）与 HALT 全部丢弃，不落架构状态。
        prog = [
            Instruction.addi(2, 0, 123),    # 0
            Instruction.addi(1, 0, 0),      # 1
            Instruction.store(2, 1, 0),     # 2 合法老 STORE
            Instruction.addi(3, 0, 2),      # 3 非对齐基址
            Instruction.load(5, 3, 0),      # 4 非法 LOAD（头部报故障）
            Instruction.addi(6, 0, 9),      # 5 年轻 ALU，可能已完成
            Instruction.halt(),             # 6
        ]
        r = run_ooo(prog, memory_from_words([0, 0]))
        seq = run_sequential(prog, memory_from_words([0, 0]))
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(r.fault_pc, 4)
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 3, 4])
        # 老 STORE 已提交，故障前内存与顺序机一致。
        self.assertEqual(words_from_memory(r.memory), [123, 0])
        self.assertEqual(r.memory, seq.memory)
        # 年轻结果被丢弃：r6 保持 0，机器未 HALT。
        self.assertEqual(r.regs[6], 0)
        self.assertFalse(r.halted)
        # 故障周期事件：fault_discard 列出窗口内全部年轻指令。
        fc = r.cycles[r.fault_cycle - 1]
        self.assertEqual(fc.retire, 4)
        self.assertIn(5, fc.fault_discard)

    def test_illegal_store_never_writes(self):
        prog = [
            Instruction.addi(1, 0, 3),
            Instruction.store(0, 1, 0),
            Instruction.halt(),
        ]
        mem = memory_from_words([7, 7])
        r = run_ooo(prog, mem)
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(words_from_memory(r.memory), [7, 7])

    def test_store_after_unresolved_branch_never_speculates(self):
        # pc2 分支依赖一条 LOAD；pc3 STORE 在分支解析前不得启动。
        # 分支实际跳到 HALT，pc3/pc4 被回收：内存必须保持初始值。
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.load(2, 1, 0),       # 1 mem[0]=0
            Instruction.beq(2, 0, 3),        # 2 0==0 -> pc5
            Instruction.store(2, 1, 0),      # 3 绝不投机写
            Instruction.addi(3, 0, 9),       # 4 被回收
            Instruction.halt(),              # 5
        ]
        r = run_ooo(prog, memory_from_words([0]))
        self.assertEqual([t.pc for t in r.trace], [0, 1, 2, 5])
        self.assertEqual(words_from_memory(r.memory), [0])
        self.assertEqual(r.regs[3], 0)
        # STORE pc3 从未执行（没有 execute 事件）。
        self.assertFalse(any(3 in c.execute for c in r.cycles))


class TestCycleEventInvariants(unittest.TestCase):
    def _check(self, r):
        """不依赖顺序机的事件结构不变量。"""
        # 周期编号从 1 连续；派发与停顿互斥；派发后窗口占用 1..ROB_SIZE。
        for i, c in enumerate(r.cycles, start=1):
            self.assertEqual(c.cycle, i)
            if c.dispatch is not None:
                self.assertIsNone(c.dispatch_stall)
                self.assertLessEqual(c.dispatch.occupancy, ROB_SIZE)
                self.assertGreaterEqual(c.dispatch.occupancy, 1)
        # 退休事件数与退休轨迹长度一致（每条恰好退休一次）。
        self.assertEqual(
            len(r.trace),
            sum(1 for c in r.cycles if c.retire is not None),
        )
        # 每次回收列出的"已完成被丢弃"项必须确实在更早周期完成过。
        done_cycles = {}
        for c in r.cycles:
            for pc in c.complete:
                done_cycles.setdefault(pc, c.cycle)
        for sq in r.squashes:
            for pc in sq.reclaimed_completed:
                self.assertIn(pc, done_cycles)
                self.assertLess(done_cycles[pc], sq.cycle)
            self.assertTrue(set(sq.reclaimed_completed).issubset(sq.squashed))

    def test_invariants_on_handful(self):
        progs = [
            [Instruction.addi(1, 0, 1), Instruction.halt()],
            [Instruction.load(1, 0, 0), Instruction.add(2, 1, 0),
             Instruction.addi(2, 2, -5), Instruction.sub(3, 2, 1),
             Instruction.store(3, 0, 4), Instruction.load(4, 0, 4),
             Instruction.halt()],
        ]
        for prog in progs:
            r = run_ooo(prog, memory_from_words([100, 0]))
            self._check(r)
            seq = run_sequential(prog, memory_from_words([100, 0]))
            self.assertEqual(r.regs, seq.regs)
            self.assertEqual(r.memory, seq.memory)


# ---------------------------------------------------------------------------
# 小规模穷举：极小字母表上 3~4 条指令的全部组合
# ---------------------------------------------------------------------------

_ALPHABET = [
    lambda: Instruction.addi(1, 0, 1),
    lambda: Instruction.addi(1, 0, 4),
    lambda: Instruction.add(1, 1, 2),
    lambda: Instruction.sub(2, 2, 1),
    lambda: Instruction.load(1, 2, 0),
    lambda: Instruction.store(1, 2, 0),
]
_BRANCH_CHOICES = [
    lambda: Instruction.beq(0, 0, 1),    # 恒真，跳到最后一条 HALT
    lambda: Instruction.beq(1, 2, 1),    # 取决于运行值
]
_EXH_INIT = [0] * 32
_EXH_INIT[2] = 4
_EXH_MEM = memory_from_words([10, 20, 30, 40])


def _effects(result, sequential=True):
    return [
        (t.pc, t.op, t.fault, t.rd, t.value, t.addr, t.mem_value)
        for t in result.trace
    ]


class TestSmallExhaustive(unittest.TestCase):
    def test_enumerate_short_programs(self):
        import itertools

        count = 0
        # 长度 3：[x, y, HALT]
        for a, b in itertools.product(range(len(_ALPHABET)), repeat=2):
            prog = [_ALPHABET[a](), _ALPHABET[b](), Instruction.halt()]
            self._compare(prog)
            count += 1
        # 长度 4：[x, y, z(可为分支), HALT]，分支 +1 静态合法（目标=HALT）
        for a, b in itertools.product(range(len(_ALPHABET)), repeat=2):
            for make in (*_ALPHABET, *_BRANCH_CHOICES):
                prog = [_ALPHABET[a](), _ALPHABET[b](), make(),
                        Instruction.halt()]
                self._compare(prog)
                count += 1
        self.assertGreaterEqual(count, 300)

    def _compare(self, prog):
        seq = run_sequential(prog, bytearray(_EXH_MEM), _EXH_INIT)
        ooo = run_ooo(prog, bytearray(_EXH_MEM), _EXH_INIT)
        self.assertEqual(ooo.halted, seq.halted)
        self.assertEqual(ooo.fault, seq.fault)
        self.assertEqual(ooo.fault_pc, seq.fault_pc)
        self.assertEqual(ooo.regs, seq.regs)
        self.assertEqual(ooo.memory, seq.memory)
        self.assertEqual(_effects(ooo), _effects(seq))


# ---------------------------------------------------------------------------
# 确定性随机差分：针对乱序窗口设计的程序生成器
# ---------------------------------------------------------------------------


def make_reorder_program(seed):
    """造"双串行 LOAD 堵门 + 依赖装入的 BEQ + 多条年轻独立 ALU"程序。

    结构（每个区块）::

        LOAD rL1, r8, 0      # 第一道堵门（3 拍延迟）
        LOAD rL2, r8, 0      # 第二道：保序门控等第一道退休后才启动
        BEQ rL2, r5, +4      # 依赖装入值；命中则回收下面 3 条年轻 ALU
        ADDI ...             # 年轻 ALU ×3（多数在分支解析前已乱序完成）
        ADDI ...
        ADDI ...
      target:
        ADDI ...             # 分支目标（命中/不命中都到达）
        （可能）STORE 安全字  # 顺序访存锻炼

    期望值只取自独立顺序解释器；mem[0] 固定 7，r5 初值随机取 7/其它
    以混合命中与不命中。寄存器循环复用制造跨区块 WAW。
    """
    rng = random.Random(seed)
    blocks = 1 + seed % 3
    prog = []
    init = [0] * 32
    init[8] = 0                       # r8 = 安全对齐基址 0
    pool = 11
    for _ in range(blocks):
        rl1, rl2 = 9, 10
        prog.append(Instruction.load(rl1, 8, 0))
        prog.append(Instruction.load(rl2, 8, 0))
        # BEQ 位于 3 条年轻 ALU 之前（程序序）；+4 跳过它们到目标。
        prog.append(Instruction.beq(rl2, 5, 4))
        for _ in range(3):
            prog.append(Instruction.addi(pool, 0, rng.randrange(-9, 10)))
            pool += 1
        treg = pool                    # 分支目标
        prog.append(Instruction.addi(treg, 0, rng.randrange(-9, 10)))
        pool += 1
        if rng.random() < 0.6:
            # 只写 mem[4]，保持 mem[0] 的装入值恒为 7。
            prog.append(Instruction.store(treg, 8, 4))
        # r5 混合 7（命中）与其它（不命中）。
        init[5] = rng.choice([7, 7, 0, 999])
        if pool > 27:
            pool = 11
    prog.append(Instruction.halt())
    return prog, init


class TestRandomDifferential(unittest.TestCase):
    def test_reorder_programs_match_sequential(self):
        reclaims_seen = 0
        stalls_seen = 0
        for seed in range(400):
            prog, init = make_reorder_program(seed)
            words = [(seed * 1103515245 + 12345 + i * 7) & 0xFFFFFFFF
                     for i in range(8)]
            words[0] = 7   # 与生成器约定一致：分支比较值 mem[0] = 7
            mem = memory_from_words(words)
            seq = run_sequential(prog, bytearray(mem), init)
            ooo = run_ooo(prog, bytearray(mem), init)
            self.assertEqual(ooo.halted, seq.halted, seed)
            self.assertEqual(ooo.fault, seq.fault, seed)
            self.assertEqual(ooo.fault_pc, seq.fault_pc, seed)
            self.assertEqual(ooo.regs, seq.regs, seed)
            self.assertEqual(ooo.memory, seq.memory, seed)
            self.assertEqual(_effects(ooo), _effects(seq), seed)
            if any(sq.reclaimed_completed for sq in ooo.squashes):
                reclaims_seen += 1
            if any(c.dispatch_stall == "rob_full" for c in ooo.cycles):
                stalls_seen += 1
        # 生成器必须真的覆盖到新机制，而不是全部退化成顺序执行。
        self.assertGreater(reclaims_seen, 20)
        self.assertGreater(stalls_seen, 20)

    def test_loop_programs_rename_state_across_repeats(self):
        # 后向循环：同一 pc 多次动态派发（标签不断增长），RAT 反复恢复。
        from test_equivalence import make_loop_program

        for seed in range(200):
            prog, init = make_loop_program(seed + 5000)
            words = [(seed * 31 + i * 17) & 0xFFFFFFFF for i in range(8)]
            mem = memory_from_words(words)
            seq = run_sequential(prog, bytearray(mem), init)
            ooo = run_ooo(prog, bytearray(mem), init)
            self.assertEqual(ooo.fault, seq.fault, seed)
            self.assertEqual(ooo.fault_pc, seq.fault_pc, seed)
            self.assertEqual(ooo.halted, seq.halted, seed)
            self.assertEqual(ooo.regs, seq.regs, seed)
            self.assertEqual(ooo.memory, seq.memory, seed)
            self.assertEqual(_effects(ooo), _effects(seq), seed)


# ---------------------------------------------------------------------------
# 输入校验与预算（与旧流水线同一静态契约）
# ---------------------------------------------------------------------------


class TestInputsAndBudget(unittest.TestCase):
    def test_static_rejections_match(self):
        with self.assertRaisesRegex(TypeError, "bool"):
            Instruction.add(True, 1, 2)  # noqa: FBT003
        with self.assertRaises(ProgramError):
            run_ooo([Instruction.addi(1, 0, 1)], MEM4)  # 无 HALT
        with self.assertRaises(ProgramError):
            run_ooo([Instruction.halt()], b"abc")
        with self.assertRaises(ProgramError):
            run_ooo([Instruction.beq(0, 0, 9), Instruction.halt()], MEM4)
        regs = [0] * 32
        regs[0] = 1
        with self.assertRaisesRegex(ProgramError, "r0 恒零"):
            run_ooo([Instruction.halt()], MEM4, regs)

    def test_failure_leaves_callers_memory_untouched(self):
        mem = bytearray(b"\x01\x02\x03\x04")
        snapshot = bytes(mem)
        with self.assertRaises(Exception):
            run_ooo([Instruction.addi(1, 0, float("inf"))], mem)  # type: ignore
        self.assertEqual(bytes(mem), snapshot)

    def test_cycle_budget_is_separate(self):
        with self.assertRaises(CycleLimitError):
            run_ooo([Instruction.beq(0, 0, 0), Instruction.halt()], MEM4)

    def test_max_program_runs_under_budget(self):
        # 200 条串行 LOAD 是吞吐最差情形之一，必须远在 2000 周期内结束。
        prog = [Instruction.addi(1, 0, 0)]
        prog += [Instruction.load(2, 1, 0)] * 198
        prog += [Instruction.halt()]
        self.assertEqual(len(prog), 200)
        r = run_ooo(prog, memory_from_words([0]))
        self.assertTrue(r.halted)
        self.assertLess(r.cycles_used, 2000)


if __name__ == "__main__":
    unittest.main()
