"""流水线内核与独立顺序参考解释器的交叉验证。

顺序解释器（tinycpu.sequential）独立实现执行语义，不调用 pipeline；
本测试用固定种子的确定性随机程序生成器造程序，期望值一律取自顺序机，
比较：退休指令序列、每条指令的架构效果、最终寄存器、异常前内存、
异常位置。另对流水线槽位做结构不变量检查（不依赖顺序机）。
"""

import random
import unittest

from tinycpu import (
    ADD,
    ADDI,
    BEQ,
    HALT,
    LOAD,
    STORE,
    SUB,
    Fault,
    Instruction,
    Op,
    memory_from_words,
)
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

MEM_WORDS = 8
MEM_BYTES = MEM_WORDS * 4


def make_program(seed: int, length: int = 14) -> tuple[list[Instruction], list[int]]:
    """生成一个保证终止（只有向前分支）的随机程序。

    基址寄存器初值混合"对齐且在内存内"与"任意值"，使访存合法/非法都出现。
    """
    rng = random.Random(seed)
    prog: list[Instruction] = []
    # 最后一条固定 HALT；前面随机生成。
    for pc in range(length - 1):
        op = rng.choices(
            [Op.ADD, Op.SUB, Op.ADDI, Op.LOAD, Op.STORE, Op.BEQ],
            weights=[3, 2, 4, 3, 3, 2],
            k=1,
        )[0]
        if op in (Op.ADD, Op.SUB):
            prog.append(
                Instruction(op, rd=rng.randrange(1, 32),
                            rs1=rng.randrange(32), rs2=rng.randrange(32))
            )
        elif op == Op.ADDI:
            prog.append(
                Instruction.addi(rng.randrange(0, 32), rng.randrange(32),
                                 rng.randrange(-40, 41))
            )
        elif op == Op.LOAD:
            prog.append(
                Instruction.load(rng.randrange(0, 32), rng.randrange(32),
                                 rng.choice([-4, 0, 0, 0, 4, 8, 2]))
            )
        elif op == Op.STORE:
            prog.append(
                Instruction.store(rng.randrange(32), rng.randrange(32),
                                  rng.choice([0, 0, 4, -4, 1, 12]))
            )
        else:  # 只生成分到"前方"的 BEQ，保证 pc 严格推进、必然终止
            max_off = length - 1 - pc
            off = rng.randrange(1, max_off + 1)
            prog.append(Instruction.beq(rng.randrange(32), rng.randrange(32), off))
    prog.append(Instruction.halt())

    init = [0] * 32
    for i in range(1, 32):
        if rng.random() < 0.6:
            init[i] = rng.randrange(0, MEM_WORDS) * 4  # 对齐合法基址
        else:
            init[i] = rng.randrange(0, 1 << 20)        # 任意值（可能致错）
    return prog, init


def make_loop_program(seed: int):
    """结构化计数循环程序（含后向 BEQ），保证终止。

    pc0           ADDI r_cnt, r0, iters
    pc1..pcK      随机循环体（不写 r_cnt；访存非法时会在循环内精确异常）
    pcK+1         ADDI r_cnt, r_cnt, -1
    pcK+2         BEQ r_cnt, r0, +2      减到 0 退出
    pcK+3         BEQ r0, r0, -(K+2)     否则回 pc1
    pcK+4         HALT
    """
    rng = random.Random(seed)
    iters = 1 + seed % 5
    cnt_reg = 31
    body_size = 2 + seed % 4
    prog = [Instruction.addi(cnt_reg, 0, iters)]
    for _ in range(body_size):
        op = rng.choice([Op.ADD, Op.SUB, Op.ADDI, Op.LOAD, Op.STORE])
        rd = rng.randrange(1, 31)
        rs1 = rng.randrange(1, 31)
        rs2 = rng.randrange(1, 31)
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
    k_plus_1 = len(prog)            # ADDI r_cnt,-1 的下标
    prog.append(Instruction.addi(cnt_reg, cnt_reg, -1))
    prog.append(Instruction.beq(cnt_reg, 0, 2))
    back = 1 - (k_plus_1 + 2)       # 使 (k+2)+back == 1
    prog.append(Instruction.beq(0, 0, back))
    prog.append(Instruction.halt())
    init = [0] * 32
    for i in range(1, 31):
        init[i] = rng.randrange(0, MEM_WORDS) * 4 if rng.random() < 0.7 else rng.randrange(0, 1 << 16)
    return prog, init


def effects_equal(t_pipe, t_seq) -> bool:
    return (
        t_pipe.pc == t_seq.pc
        and t_pipe.op == t_seq.op
        and t_pipe.fault == t_seq.fault
        and t_pipe.rd == t_seq.rd
        and t_pipe.value == t_seq.value
        and t_pipe.addr == t_seq.addr
        and t_pipe.mem_value == t_seq.mem_value
    )


class TestEquivalence(unittest.TestCase):
    def test_many_random_programs(self):
        cases = 0
        for seed in range(400):
            length = 6 + seed % 12
            prog, init = make_program(seed, length)
            words = [(seed * 7 + i * 13) & 0xFFFFFFFF for i in range(MEM_WORDS)]
            mem = memory_from_words(words)

            seq = run_sequential(prog, mem, init)
            pipe = run(prog, bytearray(mem), init)
            cases += 1

            # 终止方式与故障位置一致
            self.assertEqual(pipe.halted, seq.halted, seed)
            self.assertEqual(pipe.fault, seq.fault, seed)
            self.assertEqual(pipe.fault_pc, seq.fault_pc, seed)

            # 最终寄存器一致（含 r0 恒零）
            self.assertEqual(pipe.regs, seq.regs, seed)
            self.assertEqual(pipe.regs[0], 0, seed)

            # 异常前/最终内存一致
            self.assertEqual(pipe.memory, seq.memory, seed)

            # 退休序列与每条指令的架构效果逐条一致
            self.assertEqual(len(pipe.trace), len(seq.trace), seed)
            for tp, ts in zip(pipe.trace, seq.trace):
                self.assertTrue(effects_equal(tp, ts), (seed, tp, ts))

            # 内存写次数一致（STORE 退休数）
            stores_pipe = [t for t in pipe.trace if t.op == Op.STORE and t.fault is None]
            stores_seq = [t for t in seq.trace if t.op == Op.STORE and t.fault is None]
            self.assertEqual(len(stores_pipe), len(stores_seq), seed)

            self._check_slot_invariants(pipe, prog)
        self.assertGreaterEqual(cases, 400)

    def test_loop_programs_with_backward_branches(self):
        for seed in range(300):
            prog, init = make_loop_program(seed)
            words = [(seed * 31 + i * 17) & 0xFFFFFFFF for i in range(MEM_WORDS)]
            mem = memory_from_words(words)
            seq = run_sequential(prog, mem, init)
            pipe = run(prog, bytearray(mem), init)
            self.assertEqual(pipe.fault, seq.fault, seed)
            self.assertEqual(pipe.fault_pc, seq.fault_pc, seed)
            self.assertEqual(pipe.halted, seq.halted, seed)
            self.assertEqual(pipe.regs, seq.regs, seed)
            self.assertEqual(pipe.memory, seq.memory, seed)
            self.assertEqual(len(pipe.trace), len(seq.trace), seed)
            for tp, ts in zip(pipe.trace, seq.trace):
                self.assertTrue(effects_equal(tp, ts), (seed, tp, ts))

    def _check_slot_invariants(self, result, prog):
        """不依赖顺序机的流水槽结构不变量。"""
    def _check_slot_invariants(self, result, prog):
        """不依赖顺序机的流水槽结构不变量。"""
        from collections import Counter

        # 1) 每条退休指令：恰好占过一次非气泡 WB 槽（无重复提交），
        #    且五个级都真实经过（停顿的 hold、冲刷后重新取指不算违规，
        #    只要求覆盖与单次退休）。
        retired = [t.pc for t in result.trace]
        wb_count = Counter()
        stage_cover: dict[int, set[str]] = {}
        for c in result.cycles:
            for s in c.slots:
                if s.index is None or s.bubble:
                    continue
                stage_cover.setdefault(s.index, set()).add(s.stage)
                if s.stage == "WB":
                    wb_count[s.index] += 1
        for pc in retired:
            self.assertEqual(wb_count[pc], 1, (pc, wb_count[pc], prog))
            self.assertEqual(stage_cover[pc], {"IF", "ID", "EX", "MEM", "WB"}, pc)

        # 2) 故障丢弃的指令永不退休（故障当周期机器即停，此判据恒成立）。
        #    冲刷/停顿的结构规则：
        #    - flush 的"受害指令"（带 index）只能成对（IF 排空时单个）出现在
        #      决策周期的 ID/IF；
        #    - 插入的气泡（index 为 None）随后沿 EX->MEM->WB 向后传播；
        #    - stall 的 hold 冻结标记只在 IF/ID。
        fault_discarded = set()
        stall_seen = {"EX": -1, "MEM": -1, "WB": -1}
        for c in result.cycles:
            victims = [
                s for s in c.slots
                if s.bubble and s.reason == "flush" and s.index is not None
            ]
            if victims:
                self.assertTrue(1 <= len(victims) <= 2)
                self.assertTrue(all(s.stage in ("IF", "ID") for s in victims))
                if len(victims) == 1:
                    self.assertEqual(victims[0].stage, "ID")
                    self.assertIsNone(c.slots[0].index)
            for s in c.slots:
                if s.bubble and s.reason == "fault":
                    fault_discarded.add(s.index)
                if s.bubble and s.reason == "stall" and s.index is None:
                    if stall_seen[s.stage] == -1:
                        stall_seen[s.stage] = c.cycle
                if s.reason == "hold":
                    self.assertIn(s.stage, ("IF", "ID"))
        self.assertFalse(fault_discarded & set(retired), prog)
        # stall 气泡按 EX -> MEM -> WB 的顺序首次出现（若存在）。
        firsts = [v for v in stall_seen.values() if v != -1]
        self.assertEqual(firsts, sorted(firsts))

        # 3) hold 永远在 IF/ID；stall/flush 气泡一旦产生只会出现在
        #    EX/MEM/WB（沿流水向后传播），不会出现在 IF/ID。
        for c in result.cycles:
            for s in c.slots:
                if s.reason == "hold":
                    self.assertIn(s.stage, ("IF", "ID"))
                if s.bubble and s.reason in ("stall", "flush") and s.index is None:
                    self.assertIn(s.stage, ("EX", "MEM", "WB"))

        # 4) 每周期恰好五个槽，顺序固定。
        for c in result.cycles:
            self.assertEqual([s.stage for s in c.slots],
                             ["IF", "ID", "EX", "MEM", "WB"])

    def test_handwritten_load_use_equivalence(self):
        prog = [
            Instruction.load(1, 0, 0),
            Instruction.add(2, 1, 0),
            Instruction.addi(2, 2, -5),
            Instruction.sub(3, 2, 1),
            Instruction.store(3, 0, 4),
            Instruction.load(4, 0, 4),
            Instruction.halt(),
        ]
        mem = memory_from_words([100, 0])
        seq = run_sequential(prog, mem)
        pipe = run(prog, bytearray(mem))
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertTrue(pipe.halted and seq.halted)
        self.assertEqual(pipe.regs[4], pipe.regs[3])


class TestNoSharedCheating(unittest.TestCase):
    """顺序参考不得直接调用被测核心（静态约束检查，作为防回归哨兵）。"""

    def test_sequential_does_not_import_pipeline(self):
        import inspect
        import tinycpu.sequential as mod

        src = inspect.getsource(mod)
        self.assertNotIn("import pipeline", src)
        self.assertNotIn("from .pipeline", src)
        self.assertNotIn("from tinycpu.pipeline", src)


if __name__ == "__main__":
    unittest.main()
