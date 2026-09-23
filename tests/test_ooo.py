"""4 项重排序窗口 + 寄存器重命名乱序核心的测试。

核对方式（不用被测实现自己生成期望值）：
* 时序用手算周期表（周期相位、窗口内容、完成/退休周期逐条钉死）；
* 架构结果（寄存器/内存/退休序列/故障位置）一律取自独立顺序参考
  ``tinycpu.sequential``（它不导入乱序核心）；
* 另用确定性随机短程序（直线 + 后向分支循环）在多个窗口大小下做差分，
  以及只依赖模型自身结构定义的不变量检查。
"""

import random
import unittest

from tinycpu import (
    REORDER_WINDOW,
    Fault,
    Instruction,
    Op,
    memory_from_words,
    words_from_memory,
)
from tinycpu.ooo import (
    MAX_WINDOW,
    MIN_WINDOW,
    run_out_of_order,
)
from tinycpu.pipeline import CycleLimitError, run as run_inorder
from tinycpu.sequential import run_sequential


def _retire_pcs(result):
    return [t.pc for t in result.trace]


def _completion_pcs(result, *, include_squashed=True):
    return [
        c.pc
        for c in result.completions
        if include_squashed or not c.squashed
    ]


class TestHandComputedTiming(unittest.TestCase):
    """手算周期表：相位顺序、窗口内容、完成/退休周期。"""

    def test_dependency_chain_timeline(self):
        # 0 ADDI r1,r0,10 ; 1 ADDI r2,r1,5 ; 2 ADD r3,r1,r2 ; 3 HALT
        prog = [
            Instruction.addi(1, 0, 10),
            Instruction.addi(2, 1, 5),
            Instruction.add(3, 1, 2),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([0]))
        self.assertEqual((r.regs[1], r.regs[2], r.regs[3]), (10, 15, 25))
        # 完成周期：每条依赖链隔一个周期（c1/c2/c3）。
        alu = {c.pc: c.cycle for c in r.completions if c.kind == "alu"}
        self.assertEqual(alu, {0: 1, 1: 2, 2: 3})
        # 退休周期：第一条 c2 退休，之后每周期一条；HALT c5。
        self.assertEqual([t.cycle for t in r.trace], [2, 3, 4, 5])
        self.assertEqual(r.cycles_used, 5)
        # c1 窗口被 4 条填满（含已完成的 pc0 与读入即完成的 HALT）。
        c1 = r.window[0]
        self.assertEqual([s.pc for s in c1.slots], [0, 1, 2, 3])
        self.assertEqual(
            [s.state for s in c1.slots], ["done", "busy", "busy", "done"]
        )

    def test_independent_young_instruction_completes_first(self):
        # 题面"整数算术乱序完成"：不依赖 LOAD 的年轻 ADDI 比 LOAD 先完成。
        # 0 LOAD r1,0(r0)（mem[0]=7，顺序访存，c1 访存相位完成）
        # 1 ADDI r3,r1,1（等 LOAD，c2 才完成）
        # 2 ADDI r4,r0,99（无依赖，c1 整数口先完成）
        # 3 HALT
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.addi(3, 1, 1),
            Instruction.addi(4, 0, 99),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([7]))
        # 完成顺序与程序序不同：pc2 -> pc0 -> pc1。
        self.assertEqual(_completion_pcs(r, include_squashed=False), [2, 0, 1])
        # 但退休严格按程序序。
        self.assertEqual(_retire_pcs(r), [0, 1, 2, 3])
        self.assertEqual((r.regs[1], r.regs[3], r.regs[4]), (7, 8, 99))
        # 完成不早于……退休不早于完成。
        complete_cycle = {c.pc: c.cycle for c in r.completions if not c.squashed}
        for t in r.trace:
            if t.op != Op.HALT:
                self.assertGreaterEqual(t.cycle, complete_cycle[t.pc])

    def test_window_full_stalls_fetch(self):
        # 5 条指令、窗口 4：c1 只能取 4 条，c2 退休一条后才补取 HALT。
        prog = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(2, 0, 2),
            Instruction.addi(3, 0, 3),
            Instruction.addi(4, 0, 4),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([0]))
        self.assertEqual(r.window[0].fetch_pc, 4)
        self.assertEqual(len(r.window[0].slots), 4)
        self.assertEqual(r.window[1].fetch_pc, 5)  # c2 取出 HALT
        for w in r.window:
            self.assertLessEqual(len(w.slots), REORDER_WINDOW)


class TestRenameAndSquash(unittest.TestCase):
    def test_waw_war_rename_independence(self):
        # WAW：r4 先被一条两拍依赖链上的老指令写，再被独立年轻指令覆盖写；
        # 老写未就绪时，年轻写在 c1 抢跑完成，后续读者经标签拿到年轻值 7。
        init = [0] * 32
        init[4] = 6
        prog = [
            Instruction.load(1, 0, 0),      # 0 r1 = mem[0] = 100（c1 访存相位）
            Instruction.addi(9, 1, 0),      # 1 c2 才完成
            Instruction.add(4, 9, 0),       # 2 老写：r4 = 100（c3 才完成）
            Instruction.addi(4, 0, 7),      # 3 年轻独立写：r4 = 7（c1 抢跑）
            Instruction.add(6, 0, 4),       # 4 经标签读 pc3 的 r4（=7）
            Instruction.sub(7, 0, 4),       # 5 同标签 -> -7 = 0xFFFFFFF9
            Instruction.addi(8, 0, 0),      # 6 占位
            Instruction.halt(),             # 7
        ]
        r = run_out_of_order(prog, memory_from_words([100]), init)
        self.assertEqual(r.regs[4], 7)          # 架构上以最后一次写为准
        self.assertEqual(r.regs[6], 7)
        self.assertEqual(r.regs[7], 0xFFFFFFF9)
        # pc3（年轻写）比 pc2（老写）先完成；退休序仍是程序序 2 在 3 前。
        cyc = {c.pc: c.cycle for c in r.completions if not c.squashed}
        self.assertLess(cyc[3], cyc[2])
        self.assertEqual(_retire_pcs(r), list(range(8)))

    def test_mispredict_squashes_completed_young_results(self):
        # 关键场景：年轻整数结果先乱序完成，分支解析错误后被回收。
        # 0 LOAD r1,0(r0)（mem[0]=5）
        # 1 BEQ r1,r1,+4 -> pc5（等 LOAD，c2 才解析；必命中）
        # 2 ADDI r4,r0,99（c1 乱序完成——推测结果）
        # 3 ADDI r5,r0,88（c2 解析前抢跑完成——推测结果）
        # 4 ADDI r7,r0,5 （c2 补取，未完成即被冲刷）
        # 5 ADDI r6,r0,7（分支目标）
        # 6 HALT
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.beq(1, 1, 4),
            Instruction.addi(4, 0, 99),
            Instruction.addi(5, 0, 88),
            Instruction.addi(7, 0, 5),
            Instruction.addi(6, 0, 7),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([5]))
        # 退休序列：错误路径上的 pc2/3/4 全部消失。
        self.assertEqual(_retire_pcs(r), [0, 1, 5, 6])
        # 最终架构状态不含被回收的年轻结果。
        self.assertEqual((r.regs[4], r.regs[5], r.regs[7]), (0, 0, 0))
        self.assertEqual(r.regs[6], 7)
        # c2 有一次重定向，冲刷名单恰好 [4,3,2]（最年轻到最老）。
        redirect_cycles = [w for w in r.window if w.redirect is not None]
        self.assertEqual(len(redirect_cycles), 1)
        w = redirect_cycles[0]
        self.assertEqual((w.redirect, w.squashed), (5, (4, 3, 2)))
        # pc2/pc3 的完成记录带 squashed 标记且带值（99/88 真实算出后被回收）。
        sq = {(c.pc, c.value) for c in r.completions if c.squashed}
        self.assertIn((2, 99), sq)
        self.assertIn((3, 88), sq)
        self.assertIn((4, None), sq)
        # 冲刷后 RAT 已还原：后续没有任何条目再等待被回收标签（程序能正常
        # 跑完即说明标签引用有效，另用顺序参考对账）。
        seq = run_sequential(prog, memory_from_words([5]))
        self.assertEqual(r.regs, seq.regs)

    def test_squashed_fetch_includes_speculative_halt(self):
        # HALT 沿不命中路径被推测读入，命中分支把它一起冲刷后仍能继续取指。
        prog = [
            Instruction.addi(1, 0, 1),    # 0
            Instruction.beq(0, 0, 2),     # 1 必命中 -> pc3（冲刷 pc2）
            Instruction.halt(),           # 2 被冲刷的 HALT
            Instruction.addi(2, 0, 42),   # 3
            Instruction.halt(),           # 4
        ]
        r = run_out_of_order(prog, memory_from_words([0]))
        self.assertTrue(r.halted)
        self.assertEqual(_retire_pcs(r), [0, 1, 3, 4])
        self.assertEqual(r.regs[2], 42)

    def test_not_taken_keeps_young_instructions(self):
        prog = [
            Instruction.addi(1, 0, 3),
            Instruction.addi(2, 0, 4),
            Instruction.beq(1, 2, 2),     # 不命中
            Instruction.addi(3, 0, 9),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([0]))
        self.assertEqual(_retire_pcs(r), [0, 1, 2, 3, 4])
        self.assertFalse(any(w.redirect is not None for w in r.window))
        self.assertFalse(any(c.squashed for c in r.completions))


class TestOrderedMemory(unittest.TestCase):
    def test_load_blocked_behind_unresolved_head(self):
        # 访存按程序序串行：后一个 LOAD 必须等前一个 LOAD 退休才执行；
        # 与访存无关的年轻算术在整数口上越过等待链先完成。
        # 0 LOAD r1,0(r0) （mem[0]=55，c1 访存相位完成）
        # 1 ADDI r2,r0,0  （零基址，c1 整数口完成）
        # 2 LOAD r4,0(r2) （c3 才成为可执行队首）
        # 3 LOAD r5,4(r2) （堵在 pc2 后，c4 才执行，读 mem[4]=77）
        # 4 ADDI r6,r0,9  （与访存无关，c2 越过等待链完成）
        # 5 HALT
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.addi(2, 0, 0),
            Instruction.load(4, 2, 0),
            Instruction.load(5, 2, 4),
            Instruction.addi(6, 0, 9),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([55, 77]))
        self.assertEqual((r.regs[1], r.regs[2], r.regs[4], r.regs[5]),
                         (55, 0, 55, 77))
        self.assertEqual(r.regs[6], 9)
        cyc = {c.pc: c.cycle for c in r.completions if not c.squashed}
        # 两个 LOAD 严格按序：pc2 在 c3，pc3 在 c4。
        self.assertEqual((cyc[2], cyc[3]), (3, 4))
        # 独立算术 pc4 在 c2 完成，早于被堵住的 pc3。
        self.assertEqual(cyc[4], 2)
        self.assertLess(cyc[4], cyc[3])
        # 退休序仍是程序序。
        self.assertEqual(_retire_pcs(r), [0, 1, 2, 3, 4, 5])

    def test_faulting_head_store_young_store_never_writes(self):
        prog = [
            Instruction.addi(5, 0, 111),   # 0
            Instruction.addi(1, 0, 0),     # 1
            Instruction.store(5, 1, 0),    # 2 合法老 STORE
            Instruction.addi(6, 0, 2),     # 3 非对齐基址
            Instruction.load(7, 6, 0),     # 4 非法 LOAD（队首执行时识别）
            Instruction.store(5, 1, 8),    # 5 年轻合法 STORE，禁止执行/写内存
            Instruction.halt(),            # 6
        ]
        r = run_out_of_order(prog, memory_from_words([0, 0, 0, 0]))
        self.assertEqual(r.fault, Fault.ADDR)
        self.assertEqual(r.fault_pc, 4)
        self.assertEqual(words_from_memory(r.memory), [111, 0, 0, 0])
        self.assertEqual(_retire_pcs(r), [0, 1, 2, 3, 4])
        # pc5 从未进入执行（顺序访存：不是队首），无完成记录。
        self.assertFalse(any(c.pc == 5 for c in r.completions))

    def test_illegal_store_after_taken_branch(self):
        prog = [
            Instruction.addi(1, 0, 4),                # 0
            Instruction.addi(2, 0, 4),                # 1
            Instruction.beq(1, 2, 2),                 # 2 -> 4
            Instruction.addi(3, 0, 9),                # 3 冲刷
            Instruction.store(3, 0, 0),               # 4 冲刷
            Instruction.addi(4, 0, 7),                # 5 非对齐
            Instruction.store(1, 4, 0),               # 6 非法 STORE
            Instruction.store(2, 0, 4),               # 7 年轻 STORE 禁写
            Instruction.halt(),                       # 8
        ]
        r = run_out_of_order(prog, memory_from_words([0, 0, 0]))
        self.assertEqual(r.fault_pc, 6)
        self.assertEqual(words_from_memory(r.memory), [0, 0, 0])
        self.assertEqual(r.regs[3], 0)

    def test_r0_store_value_via_rename(self):
        # STORE 的数据经重命名标签取得（ADD 结果），地址经另一标签取得。
        prog = [
            Instruction.addi(1, 0, 4),
            Instruction.addi(2, 0, 0x41),
            Instruction.add(3, 2, 0),
            Instruction.store(3, 1, 0),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, memory_from_words([0, 0]))
        # 地址 4（字节）= 第 2 个字。
        self.assertEqual(words_from_memory(r.memory), [0, 0x41])


class TestWindowParameter(unittest.TestCase):
    def _prog(self):
        return [
            Instruction.addi(1, 0, 1),
            Instruction.add(2, 1, 1),
            Instruction.halt(),
        ]

    def test_window_one_is_in_order(self):
        prog = self._prog()
        r = run_out_of_order(prog, memory_from_words([0]), window=1)
        # 窗口 1：无乱序空间，完成顺序 == 程序序。
        self.assertEqual(
            _completion_pcs(r, include_squashed=False), [0, 1]
        )
        self.assertEqual(_retire_pcs(r), [0, 1, 2])
        self.assertEqual(r.regs[2], 2)
        for w in r.window:
            self.assertLessEqual(len(w.slots), 1)

    def test_window_sizes_all_match_sequential(self):
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.addi(2, 1, 3),
            Instruction.add(3, 1, 2),
            Instruction.beq(2, 3, 2),
            Instruction.addi(4, 0, 99),    # 2==? 不命中时执行
            Instruction.store(3, 0, 4),
            Instruction.halt(),
        ]
        mem = memory_from_words([10, 0])
        seq = run_sequential(prog, bytearray(mem))
        for w in range(MIN_WINDOW, MAX_WINDOW + 1):
            r = run_out_of_order(prog, bytearray(mem), window=w)
            self.assertEqual(r.regs, seq.regs, w)
            self.assertEqual(r.memory, seq.memory, w)
            self.assertEqual(_retire_pcs(r), [t.pc for t in seq.trace], w)

    def test_window_validation(self):
        prog = [Instruction.halt()]
        mem = memory_from_words([0])
        for bad in (0, MAX_WINDOW + 1, -1):
            with self.assertRaisesRegex(ValueError, "window"):
                run_out_of_order(prog, mem, window=bad)
        with self.assertRaisesRegex(TypeError, "bool"):
            run_out_of_order(prog, mem, window=True)  # noqa: FBT003
        with self.assertRaisesRegex(TypeError, "window"):
            run_out_of_order(prog, mem, window=2.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "window"):
            run_out_of_order(prog, mem, window="4")  # type: ignore[arg-type]

    def test_default_window_is_four(self):
        self.assertEqual(REORDER_WINDOW, 4)


class TestDifferential(unittest.TestCase):
    """确定性随机短程序：乱序核心逐退休对照独立顺序模型。"""

    def _assert_matches_sequential(self, prog, init, mem, window, seed):
        seq = run_sequential(prog, bytearray(mem), init)
        ooo = run_out_of_order(prog, bytearray(mem), init, window=window)
        self.assertEqual(ooo.halted, seq.halted, seed)
        self.assertEqual(ooo.fault, seq.fault, seed)
        self.assertEqual(ooo.fault_pc, seq.fault_pc, seed)
        self.assertEqual(ooo.regs, seq.regs, seed)
        self.assertEqual(ooo.regs[0], 0, seed)
        self.assertEqual(ooo.memory, seq.memory, seed)
        self.assertEqual(len(ooo.trace), len(seq.trace), seed)
        for tp, ts in zip(ooo.trace, seq.trace):
            self.assertEqual(
                (tp.pc, tp.op, tp.fault, tp.rd, tp.value, tp.addr, tp.mem_value),
                (ts.pc, ts.op, ts.fault, ts.rd, ts.value, ts.addr, ts.mem_value),
                seed,
            )
        # 结构不变量：窗口不超限；退休严格在完成之后；窗口条目标签单调。
        for w in ooo.window:
            self.assertLessEqual(len(w.slots), window, seed)
            self.assertEqual(
                [s.tag for s in w.slots],
                sorted(s.tag for s in w.slots),
                seed,
            )
        # 每条退休的非 HALT 指令都有非冲刷完成记录，且完成周期 <= 退休周期。
        first_complete = {}
        for c in ooo.completions:
            if not c.squashed:
                first_complete.setdefault(c.pc, c.cycle)
        for t in ooo.trace:
            if t.op != Op.HALT:
                self.assertIn(t.pc, first_complete, seed)
                self.assertLessEqual(first_complete[t.pc], t.cycle, seed)
        # 被冲刷的完成：当周期窗口末态中这些 pc 已不存在。
        for c in ooo.completions:
            if c.squashed:
                snap = ooo.window[c.cycle - 1]
                self.assertNotIn(c.pc, [s.pc for s in snap.slots], seed)

    def test_straight_line_programs(self):
        rng = random.Random(20260924)
        cases = 0
        for seed in range(240):
            length = 5 + seed % 10
            prog = self._make_straight(rng, length)
            init = self._make_init(rng)
            words = [(seed * 7 + i * 13) & 0xFFFFFFFF for i in range(8)]
            mem = memory_from_words(words)
            window = (1, 2, 4, 8)[seed % 4]
            self._assert_matches_sequential(prog, init, mem, window, seed)
            cases += 1
        self.assertGreaterEqual(cases, 240)

    def test_loop_programs(self):
        for seed in range(180):
            prog, init = self._make_loop(seed)
            words = [(seed * 31 + i * 17) & 0xFFFFFFFF for i in range(8)]
            mem = memory_from_words(words)
            window = (1, 3, 4, 7)[seed % 4]
            self._assert_matches_sequential(prog, init, mem, window, ("loop", seed))

    def test_legal_memory_heavy_programs(self):
        # 全部使用对齐合法地址：迫使 STORE/LOAD 在乱序算术间反复按序串行。
        rng = random.Random(4242)
        for seed in range(120):
            prog = [
                Instruction.addi(1, 0, 0),
                Instruction.addi(2, 0, 0),
            ]
            for _ in range(4 + seed % 6):
                kind = rng.randrange(5)
                if kind == 0:
                    prog.append(Instruction.add(rng.randrange(3, 10),
                                                rng.randrange(1, 10),
                                                rng.randrange(1, 10)))
                elif kind == 1:
                    prog.append(Instruction.addi(rng.randrange(3, 10),
                                                 rng.randrange(1, 10),
                                                 rng.randrange(-8, 9)))
                elif kind == 2:
                    prog.append(Instruction.load(rng.randrange(3, 10), 1,
                                                 rng.choice([0, 4, 8, 12, 16, 20, 24])))
                elif kind == 3:
                    prog.append(Instruction.store(rng.randrange(2, 10), 1,
                                                  rng.choice([0, 4, 8, 12, 16, 20, 24])))
                else:
                    prog.append(Instruction.sub(rng.randrange(3, 10),
                                                rng.randrange(1, 10),
                                                rng.randrange(1, 10)))
            prog.append(Instruction.halt())
            init = [0] * 32
            mem = memory_from_words([(seed + i) & 0xFFFFFFFF for i in range(8)])
            self._assert_matches_sequential(prog, init, mem, 4, ("mem", seed))

    # -- 程序生成器（保证终止；期望值不在这里算，只负责造结构） ----------

    def _make_straight(self, rng, length):
        prog = []
        for pc in range(length - 1):
            op = rng.choices(
                [Op.ADD, Op.SUB, Op.ADDI, Op.LOAD, Op.STORE, Op.BEQ],
                weights=[3, 2, 4, 3, 3, 2],
                k=1,
            )[0]
            if op in (Op.ADD, Op.SUB):
                prog.append(Instruction(op, rd=rng.randrange(0, 32),
                                        rs1=rng.randrange(32), rs2=rng.randrange(32)))
            elif op == Op.ADDI:
                prog.append(Instruction.addi(rng.randrange(0, 32), rng.randrange(32),
                                             rng.randrange(-40, 41)))
            elif op == Op.LOAD:
                prog.append(Instruction.load(rng.randrange(0, 32), rng.randrange(32),
                                             rng.choice([-4, 0, 0, 0, 4, 8, 2])))
            elif op == Op.STORE:
                prog.append(Instruction.store(rng.randrange(32), rng.randrange(32),
                                              rng.choice([0, 0, 4, -4, 1, 12])))
            else:
                off = rng.randrange(1, length - pc)
                prog.append(Instruction.beq(rng.randrange(32), rng.randrange(32), off))
        prog.append(Instruction.halt())
        return prog

    def _make_loop(self, seed):
        rng = random.Random(seed + 7)
        iters = 1 + seed % 5
        cnt = 31
        body = 2 + seed % 4
        prog = [Instruction.addi(cnt, 0, iters)]
        for _ in range(body):
            op = rng.choice([Op.ADD, Op.SUB, Op.ADDI, Op.LOAD, Op.STORE])
            rd, rs1, rs2 = (rng.randrange(1, 31) for _ in range(3))
            if op == Op.ADD:
                prog.append(Instruction.add(rd, rs1, rs2))
            elif op == Op.SUB:
                prog.append(Instruction.sub(rd, rs1, rs2))
            elif op == Op.ADDI:
                prog.append(Instruction.addi(rd, rs1, rng.randrange(-16, 17)))
            elif op == Op.LOAD:
                prog.append(Instruction.load(rd, rs1, rng.choice([0, 4, -4, 2])))
            else:
                prog.append(Instruction.store(rs2, rs1, rng.choice([0, 4, 8])))
        k1 = len(prog)
        prog.append(Instruction.addi(cnt, cnt, -1))
        prog.append(Instruction.beq(cnt, 0, 2))
        prog.append(Instruction.beq(0, 0, 1 - (k1 + 2)))
        prog.append(Instruction.halt())
        init = [0] * 32
        for i in range(1, 31):
            init[i] = (
                rng.randrange(0, 8) * 4 if rng.random() < 0.7
                else rng.randrange(0, 1 << 16)
            )
        return prog, init

    def _make_init(self, rng):
        init = [0] * 32
        for i in range(1, 32):
            if rng.random() < 0.6:
                init[i] = rng.randrange(0, 8) * 4
            else:
                init[i] = rng.randrange(0, 1 << 20)
        return init


class TestExhaustiveEnumeration(unittest.TestCase):
    """小规模全枚举：2 条指令的所有模板组合 × 初值配置 × 窗口大小。

    指令模板限定在 3 个寄存器、立即数 {-1,0,1}、访存偏移 {0,4}，
    程序必为直线（无分支）且以 HALT 结尾；覆盖 WAW/WAR/RAW、
    合法与非对齐/越界访存。期望值只取自顺序参考。
    """

    def _templates(self):
        t = []
        for rd in (1, 2):
            for rs in (0, 1, 2):
                for imm in (-1, 0, 1):
                    t.append(Instruction.addi(rd, rs, imm))
        for rd in (1, 2):
            for rs1 in (0, 1, 2):
                for rs2 in (0, 1, 2):
                    t.append(Instruction.add(rd, rs1, rs2))
        for rd in (1, 2):
            for rs1 in (0, 1, 2):
                for off in (0, 4):
                    t.append(Instruction.load(rd, rs1, off))
        for rs1 in (0, 1, 2):
            for rs2 in (0, 1, 2):
                for off in (0, 4):
                    t.append(Instruction.store(rs2, rs1, off))
        return t

    def test_all_two_instruction_programs(self):
        templates = self._templates()
        mem = memory_from_words([0xA5, 0x5A])
        inits = []
        for a, b in ((0, 0), (4, 0), (2, 4), (8, 4)):
            v = [0] * 32
            v[1], v[2] = a, b
            inits.append(v)

        count = 0
        for i0 in templates:
            for i1 in templates:
                prog = [i0, i1, Instruction.halt()]
                for init in inits:
                    seq = run_sequential(prog, bytearray(mem), init)
                    for w in (1, 2, 4):
                        ooo = run_out_of_order(
                            prog, bytearray(mem), init, window=w
                        )
                        self.assertEqual(ooo.regs, seq.regs,
                                         (prog, init, w))
                        self.assertEqual(ooo.memory, seq.memory,
                                         (prog, init, w))
                        self.assertEqual(ooo.fault, seq.fault,
                                         (prog, init, w))
                        self.assertEqual(ooo.fault_pc, seq.fault_pc,
                                         (prog, init, w))
                        self.assertEqual(_retire_pcs(ooo),
                                         [t.pc for t in seq.trace],
                                         (prog, init, w))
                        count += 1
        self.assertGreater(count, 30000)


class TestSharedContractAndGuards(unittest.TestCase):
    def test_same_static_validation_as_inorder(self):
        # 乱序入口复用同一套静态契约：同样非法、同样的错误类型。
        with self.assertRaises(ValueError):
            run_out_of_order(
                [Instruction.addi(1, 0, 1)], memory_from_words([0])
            )  # 不以 HALT 结尾
        mem = bytearray(b"\x01\x02\x03\x04")
        snapshot = bytes(mem)
        with self.assertRaises(TypeError):
            run_out_of_order(
                [Instruction.add(True, 1, 2), Instruction.halt()], mem
            )
        self.assertEqual(bytes(mem), snapshot)  # 失败路径不留半更新

    def test_infinite_loop_hits_budget(self):
        prog = [Instruction.beq(0, 0, 0), Instruction.halt()]
        with self.assertRaises(CycleLimitError):
            run_out_of_order(prog, memory_from_words([0]))

    def test_result_is_detached_from_inputs(self):
        mem = bytearray(memory_from_words([1, 2]))
        prog = [
            Instruction.addi(1, 0, 0),
            Instruction.store(1, 1, 0),
            Instruction.halt(),
        ]
        r = run_out_of_order(prog, mem)
        mem[0:4] = (999).to_bytes(4, "little")
        self.assertEqual(words_from_memory(r.memory), [0, 2])

    def test_old_entry_still_available(self):
        # 旧入口与行为不变（抽查：旧流水线周期数仍为 8）。
        prog = [
            Instruction.addi(1, 0, 5),
            Instruction.addi(2, 0, 7),
            Instruction.add(3, 1, 2),
            Instruction.halt(),
        ]
        r = run_inorder(prog, memory_from_words([0]))
        self.assertEqual(r.cycles_used, 8)
        self.assertEqual(r.regs[3], 12)


if __name__ == "__main__":
    unittest.main()
