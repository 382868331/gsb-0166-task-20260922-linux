"""显式外部中断与 IRET 的针对性测试与差分测试。

事件注入对齐方法：
* 流水线用 :class:`tinycpu.InterruptController` 按*周期*给脉冲；
* 顺序参考用 :class:`tinycpu.BoundaryInterrupts` 按*退休边界*（此前已退休
  指令数）给脉冲；
* 先无中断干跑流水线，得到各退休指令的周期号，把"某指令退休后的边界"
  同时映射成周期脉冲与边界脉冲——两个执行器于是在**同一退休边界**
  观察到同一事件，再比较最终寄存器与内存（期望值只取自独立顺序机）。

程序布局约定（固定向量 VECTOR_PC=0，冷启动也从 pc0 起，用 r30 区分）：
    pc0   BEQ r30, r0, +1      r30==0（中断进入）-> pc2 处理例程
    pc1   BEQ r0, r0, +(H-1)  冷启动（r30=1）-> main 入口 pcH
    pc2.. 处理例程 ...
          IRET
    pcH   ADDI r30, r0, 0     main 第一条：清冷启动标志
          ... main ...
          HALT
"""

import unittest

from tinycpu import (
    VECTOR_PC,
    BoundaryInterrupts,
    Instruction,
    InterruptController,
    InterruptStateError,
    memory_from_words,
    words_from_memory,
)
from tinycpu.isa import MAX_CYCLES
from tinycpu.pipeline import run
from tinycpu.sequential import run_sequential

# 冷启动标志寄存器 / 处理例程 scratch 寄存器
FLAG = 30
SCR = 20


def build_program(main, handler=None):
    """组装 [pc0 引导, pc1 冷启动跳转, handler(含 IRET), main(含 HALT)]。

    返回 (prog, main_start)。handler 为指令列表（不含 IRET）。
    main 第一条自动插入清标志指令。
    """
    if handler is None:
        handler = [Instruction.addi(SCR, SCR, 1)]
    h = 2 + len(handler) + 1  # handler 体 + 末尾 IRET 之后即 main 入口
    prog = [
        Instruction.beq(FLAG, 0, 2),       # pc0 中断入口 -> pc2
        Instruction.beq(0, 0, h - 1),      # pc1 冷启动 -> pch
    ]
    prog.extend(handler)
    prog.append(Instruction.iret())
    assert len(prog) == h
    prog.append(Instruction.addi(FLAG, 0, 0))
    prog.extend(main)
    return prog, h


def initial_regs():
    regs = [0] * 32
    regs[FLAG] = 1
    return regs


def retire_cycle_map(dry):
    """无中断干跑结果：{指令下标: 退休周期}（每条指令最后一次退休）。"""
    return {t.pc: t.cycle for t in dry.trace if t.fault is None}


def boundary_index(dry, pc):
    """指令 pc 退休的边界编号 = 它退休前已退休的指令数。"""
    pcs = [t.pc for t in dry.trace if t.fault is None]
    return pcs.index(pc)


class TestInterruptBasics(unittest.TestCase):
    def setUp(self):
        # main：纯直线加法，最后 HALT
        self.main = [
            Instruction.addi(1, 0, 10),       # h
            Instruction.addi(2, 0, 20),       # h+1
            Instruction.add(3, 1, 2),         # h+2
            Instruction.addi(4, 0, 99),       # h+3
            Instruction.halt(),               # h+4
        ]
        self.prog, self.h = build_program(self.main)
        self.mem = memory_from_words([0] * 4)

    def test_basic_accept_save_flush_iret(self):
        # 脉冲在 main 第一条（清标志，pc=h）退休的边界：
        # 保存 h+1，转 pc0，处理例程 r20+=1，IRET 回 h+1 继续。
        dry = run(self.prog, bytearray(self.mem), initial_regs())
        cmap = retire_cycle_map(dry)
        tcycle = cmap[self.h]
        tb = boundary_index(dry, self.h)

        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)

        pipe = run(self.prog, bytearray(self.mem), initial_regs(), ic)
        seq = run_sequential(self.prog, bytearray(self.mem), initial_regs(), bc)

        self.assertEqual(len(pipe.interrupts), 1)
        acc = pipe.interrupts[0]
        self.assertEqual(acc.cycle, tcycle)
        self.assertEqual(acc.retired_pc, self.h)
        self.assertEqual(acc.saved_pc, self.h + 1)
        # 接纳轨迹挂在该退休记录上
        rec = next(t for t in pipe.trace if t.event == "INTR")
        self.assertEqual((rec.pc, rec.saved_pc), (self.h, self.h + 1))
        # IRET 记录恢复点
        iret_rec = next(t for t in pipe.trace if t.event == "IRET")
        self.assertEqual(iret_rec.saved_pc, self.h + 1)
        # 处理例程效果与 main 现场都保留
        self.assertEqual(pipe.regs[SCR], 1)
        self.assertEqual((pipe.regs[1], pipe.regs[2], pipe.regs[3]), (10, 20, 30))
        self.assertTrue(pipe.halted and pipe.fault is None)
        self.assertFalse(pipe.masked and pipe.interrupts_pending)
        # 与独立顺序参考：寄存器 / 内存 / 退休序列全一致
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(
            [(t.pc, t.event, t.saved_pc) for t in pipe.trace],
            [(t.pc, t.event, t.saved_pc) for t in seq.trace],
        )
        self.assertEqual(
            [a.saved_pc for a in pipe.interrupts],
            [a.saved_pc for a in seq.interrupts],
        )

    def test_saved_pc_skips_branch_target_flush(self):
        # 分支在 EX 决策、年轻指令被冲刷的那个退休边界注入：
        # 保存点必须是分支目标，被冲刷的受害指令不算"尚未退休"。
        # 布局：h=清标志；main 从 h+1 起。
        main = [
            Instruction.addi(1, 0, 3),          # h+1
            Instruction.addi(2, 0, 3),          # h+2
            Instruction.beq(1, 2, 2),           # h+3 -> h+5
            Instruction.addi(9, 0, 123),        # h+4 被冲刷
            Instruction.addi(4, 0, 7),          # h+5 分支目标
            Instruction.halt(),                 # h+6
        ]
        prog, h = build_program(main)
        dry = run(prog, bytearray(self.mem), initial_regs())
        cmap = retire_cycle_map(dry)
        # 脉冲在 BEQ 自己退休的边界（此刻 IF 已重新取到分支目标 h+5）
        tcycle = cmap[h + 3]
        tb = boundary_index(dry, h + 3)

        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(self.mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(self.mem), initial_regs(), bc)

        self.assertEqual(pipe.interrupts[0].saved_pc, h + 5)
        # 被冲刷的 h+4 永不退休（中断恢复后从 h+5 走）
        self.assertNotIn(h + 4, [t.pc for t in pipe.trace])
        self.assertEqual(pipe.regs[9], 0)
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)

    def test_flushed_young_store_does_not_write(self):
        # 接纳边界上一条年轻合法 STORE 正在 MEM：冲刷后它不得写内存；
        # 分支目标越过它，恢复后也不再执行。
        main = [
            Instruction.addi(1, 0, 0),           # h  基址 0
            Instruction.addi(5, 0, 77),          # h+1
            Instruction.beq(0, 0, 3),            # h+2 -> h+5（越过 STORE）
            Instruction.store(5, 1, 0),          # h+3 被冲刷/被中断冲刷，永不执行
            Instruction.addi(9, 0, 1),           # h+4 同样越过
            Instruction.addi(6, 0, 8),           # h+5
            Instruction.halt(),                  # h+6
        ]
        prog, h = build_program(main)
        dry = run(prog, bytearray(self.mem), initial_regs())
        cmap = retire_cycle_map(dry)
        # h+1（ADDI r5）退休的周期：BEQ 在 EX、STORE 尚未取到；
        # 改在 h+2（BEQ）退休边界：STORE 在更年轻级，随冲刷丢弃且
        # 保存点是分支目标 h+5 -> STORE 永不退休。
        tcycle = cmap[h + 2]
        tb = boundary_index(dry, h + 2)
        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(self.mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(self.mem), initial_regs(), bc)

        stores = [t for t in pipe.trace if t.op.value == "STORE"]
        self.assertEqual(stores, [])
        self.assertEqual(words_from_memory(pipe.memory), [0, 0, 0, 0])
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(pipe.regs, seq.regs)

    def test_nested_interrupt_masked_and_coalesced(self):
        # 第一次中断在 h 边界接纳；仅在接纳后的头两拍/头两个边界继续给
        # 脉冲（此时必在屏蔽期）：多个脉冲合并成一位 pending。IRET 解除
        # 屏蔽后，恢复流第一条指令退休边界凭这一位恰好再接纳一次；
        # 此后再无脉冲 -> 处理例程共执行 2 次，r20=2，结束时无 pending。
        main = [
            Instruction.addi(1, 0, 1),
            Instruction.addi(1, 1, 1),
            Instruction.addi(1, 1, 1),
            Instruction.halt(),
        ]
        prog, h = build_program(main)
        dry = run(prog, bytearray(self.mem), initial_regs())
        tcycle = retire_cycle_map(dry)[h]
        tb = boundary_index(dry, h)

        ic = InterruptController()
        ic.request(tcycle)
        ic.request(tcycle + 1)   # 屏蔽期脉冲 1
        ic.request(tcycle + 2)   # 屏蔽期脉冲 2（与上一个合并）
        bc = BoundaryInterrupts()
        bc.request_after(tb)
        bc.request_after(tb + 1)
        bc.request_after(tb + 2)

        pipe = run(prog, bytearray(self.mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(self.mem), initial_regs(), bc)

        self.assertEqual(len(pipe.interrupts), 2)
        self.assertEqual(pipe.regs[SCR], 2)
        # 第二次接纳发生在恢复后第一条 main 指令（h+1）退休边界
        self.assertEqual(pipe.interrupts[1].retired_pc, h + 1)
        self.assertEqual(pipe.interrupts[1].saved_pc, h + 2)
        self.assertTrue(pipe.halted)
        self.assertFalse(pipe.interrupts_pending)
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(
            [t.event for t in pipe.trace if t.event],
            [t.event for t in seq.trace if t.event],
        )

    def test_halt_retire_rejects_interrupt(self):
        # HALT 退休拍给脉冲：脉冲先并入 pending，但 HALT 退休后不接纳，
        # 机器直接停机；不转向量、无接纳记录。
        main = [Instruction.addi(1, 0, 1), Instruction.halt()]
        prog, h = build_program(main)
        # h=清标志；ADDI 在 h+1；HALT 在 h+2
        dry = run(prog, bytearray(self.mem), initial_regs())
        tcycle = retire_cycle_map(dry)[h + 2]
        tb = boundary_index(dry, h + 2)
        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(self.mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(self.mem), initial_regs(), bc)
        self.assertTrue(pipe.halted)
        self.assertEqual(pipe.interrupts, ())
        # 脉冲未被消费（机器已停），但绝没有转向量：退休序列止于 HALT
        self.assertTrue(pipe.interrupts_pending)
        self.assertEqual(pipe.trace[-1].op.value, "HALT")
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(seq.interrupts, ())

    def test_iret_without_active_interrupt_is_error(self):
        # 程序直接执行 IRET（无中断、未屏蔽）：明确报错。
        prog = [Instruction.iret(), Instruction.halt()]
        with self.assertRaises(InterruptStateError):
            run(prog, bytearray(self.mem))
        with self.assertRaises(InterruptStateError):
            run_sequential(prog, bytearray(self.mem))


class TestExceptionAndInterruptSameBoundary(unittest.TestCase):
    def test_exception_takes_priority_interrupt_stays_pending(self):
        # pc1 非对齐 LOAD 在 WB 报 ADDR 的同一拍给中断脉冲：
        # 异常优先，立即停机；脉冲未被消费，结果中 pending 仍为 True。
        prog = [
            Instruction.addi(1, 0, 2),
            Instruction.load(2, 1, 0),       # 地址 2 非对齐
            Instruction.addi(3, 0, 5),
            Instruction.halt(),
        ]
        mem = memory_from_words([0, 0])
        dry = run(prog, bytearray(mem))
        # 故障指令 pc1 在 c6 退休（报故）；边界编号 = 1。
        self.assertEqual(dry.fault_cycle, 6)
        ic = InterruptController().request(6)
        bc = BoundaryInterrupts().request_after(1)
        pipe = run(prog, bytearray(mem), interrupts=ic)
        seq = run_sequential(prog, bytearray(mem), interrupts=bc)

        self.assertIsNotNone(pipe.fault)
        self.assertEqual(pipe.fault_pc, 1)
        self.assertEqual(pipe.interrupts, ())
        self.assertTrue(pipe.interrupts_pending)
        # 没有转向量：退休轨迹停在故障点
        self.assertEqual([t.pc for t in pipe.trace], [0, 1])
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(pipe.fault_pc, seq.fault_pc)
        self.assertTrue(seq.interrupts_pending)

    def test_pulse_during_masked_then_fault_keeps_pending(self):
        # 处理例程自身触发非法访存；接纳后屏蔽期持续给脉冲（含故障同拍）：
        # 异常优先停机，屏蔽与 pending 都保持；与顺序机一致。
        # pc0 BEQ r30,r0,+1      中断入口 -> pc2
        # pc1 BEQ r0,r0,+4       冷启动 -> pc5
        # pc2 ADDI r1,r0,3       handler: 非对齐基址
        # pc3 LOAD r5,0(r1)      非对齐 -> WB 报 ADDR
        # pc4 IRET
        # pc5 ADDI r30,r0,0      main 起：清冷启动标志
        # pc6 ADDI r2,r0,9
        # pc7 HALT
        prog = [
            Instruction.beq(FLAG, 0, 2),
            Instruction.beq(0, 0, 4),
            Instruction.addi(1, 0, 3),
            Instruction.load(5, 1, 0),
            Instruction.iret(),
            Instruction.addi(FLAG, 0, 0),
            Instruction.addi(2, 0, 9),
            Instruction.halt(),
        ]
        mem = memory_from_words([0, 0])
        dry = run(prog, bytearray(mem), initial_regs())
        self.assertTrue(dry.halted)  # 无中断时 handler 被冷启动跳过
        tcycle = retire_cycle_map(dry)[5]
        tb = boundary_index(dry, 5)

        ic = InterruptController()
        for c in range(tcycle, MAX_CYCLES + 1):
            ic.request(c)
        bc = BoundaryInterrupts()
        for k in range(tb, 200):
            bc.request_after(k)

        pipe = run(prog, bytearray(mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(mem), initial_regs(), bc)
        self.assertIsNotNone(pipe.fault)
        self.assertEqual(pipe.fault_pc, 3)
        self.assertTrue(pipe.masked)            # 故障发生在屏蔽期
        self.assertTrue(pipe.interrupts_pending)
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(pipe.fault_pc, seq.fault_pc)
        self.assertTrue(seq.masked and seq.interrupts_pending)


class TestLoadUseBoundary(unittest.TestCase):
    def test_pulse_at_load_use_retire_boundary(self):
        # LOAD -> 使用其结果的 ADD（load-use 停一拍）。
        # 脉冲在 LOAD 退休边界（恰好停顿解除拍）：保存点必须是
        # 被冻结在 ID 的消费者 ADD，而不是别的年轻指令。
        main = [
            Instruction.addi(1, 0, 0),          # h 基址
            Instruction.load(2, 1, 0),          # h+1 LOAD
            Instruction.add(3, 2, 0),           # h+2 load-use 消费者
            Instruction.addi(4, 3, 1),          # h+3
            Instruction.halt(),                 # h+4
        ]
        prog, h = build_program(main)
        words = [42, 0, 0, 0]
        mem = memory_from_words(words)
        dry = run(prog, bytearray(mem), initial_regs())

        # 结构断言：流水线确实在 LOAD 后插了一个 stall 气泡
        stalls = [
            s for c in dry.cycles for s in c.slots
            if s.bubble and s.reason == "stall" and s.stage == "EX"
        ]
        self.assertEqual(len(stalls), 1)

        tcycle = retire_cycle_map(dry)[h + 1]   # LOAD 退休周期
        tb = boundary_index(dry, h + 1)
        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(mem), initial_regs(), bc)

        self.assertEqual(pipe.interrupts[0].saved_pc, h + 2)
        # 最终语义：LOAD 值 42 仍到达，r3=42, r4=43
        self.assertEqual((pipe.regs[2], pipe.regs[3], pipe.regs[4]), (42, 42, 43))
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        # 接纳周期的槽位：LOAD 在 WB 正常退休，年轻级标 intr 丢弃
        acc_cycle = pipe.cycles[pipe.interrupts[0].cycle - 1]
        self.assertEqual(acc_cycle.slots[4].index, h + 1)
        victims = [s for s in acc_cycle.slots[:4] if s.reason == "intr"]
        self.assertTrue(victims)


class TestAdditionalBoundaries(unittest.TestCase):
    def test_flushed_young_store_rewritten_once_after_resume(self):
        # 接纳边界上一条合法年轻 STORE 正在 MEM：冲刷时绝不写；
        # IRET 恢复后它从保存点重新执行，恰好写一次（不多不少）。
        main = [
            Instruction.addi(1, 0, 0),           # h+1 基址 0
            Instruction.addi(5, 0, 77),          # h+2
            Instruction.store(5, 1, 0),          # h+3 合法 STORE
            Instruction.addi(6, 0, 8),           # h+4
            Instruction.halt(),                  # h+5
        ]
        prog, h = build_program(main)
        mem = memory_from_words([0, 0])
        dry = run(prog, bytearray(mem), initial_regs())
        # 脉冲在 h+2（ADDI r5）退休边界：此时 h+3 STORE 在 EX/MEM 附近，
        # 被冲刷；保存点必须是 h+3。
        tcycle = retire_cycle_map(dry)[h + 2]
        tb = boundary_index(dry, h + 2)
        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(mem), initial_regs(), bc)

        self.assertEqual(pipe.interrupts[0].saved_pc, h + 3)
        # STORE 只退休一次（恢复后），值 77 恰好落一次
        stores = [t for t in pipe.trace if t.op.value == "STORE" and t.fault is None]
        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0].mem_value, 77)
        self.assertEqual(words_from_memory(pipe.memory), [77, 0])
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(
            [t.pc for t in pipe.trace if t.op.value == "STORE"],
            [t.pc for t in seq.trace if t.op.value == "STORE"],
        )

    def test_early_pulse_waiting_before_first_retire(self):
        # 脉冲在流水线还没退休任何指令的填充期（c1..c4）到达：
        # 合并为一位 pending，在第一条指令正常退休（c5）边界即被接纳。
        main = [
            Instruction.addi(1, 0, 11),
            Instruction.addi(2, 0, 22),
            Instruction.halt(),
        ]
        prog, h = build_program(main)
        ic = InterruptController()
        for c in range(1, 5):
            ic.request(c)
        bc = BoundaryInterrupts()
        bc.request_after(0)  # 第一条指令退休边界
        pipe = run(prog, bytearray(memory_from_words([0])), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(memory_from_words([0])),
                             initial_regs(), bc)
        self.assertEqual(len(pipe.interrupts), 1)
        self.assertEqual(pipe.interrupts[0].cycle, 5)
        self.assertEqual(pipe.interrupts[0].retired_pc, 0)  # 冷启动后退休的 pc0
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)

    def test_pulse_same_cycle_as_iret_retire_not_accepted_until_resume(self):
        # IRET 退休拍到达的脉冲：IRET 边界不接纳；屏蔽解除后，恢复流
        # 第一条指令退休边界凭 pending 再次进入向量。用覆盖 IRET 拍的
        # 脉冲窗口注入（屏蔽期脉冲合并成一位），两侧窗口对齐到退休边界。
        main = [
            Instruction.addi(1, 0, 5),
            Instruction.addi(1, 1, 1),
            Instruction.halt(),
        ]
        prog, h = build_program(main)
        dry = run(prog, bytearray(memory_from_words([0])), initial_regs())
        tcycle = retire_cycle_map(dry)[h + 1]
        tb = boundary_index(dry, h + 1)
        first = run(
            prog, bytearray(memory_from_words([0])), initial_regs(),
            InterruptController().request(tcycle),
        )
        iret_cycle = next(t.cycle for t in first.trace if t.event == "IRET")

        ic = InterruptController()
        for c in range(tcycle, iret_cycle + 3):
            ic.request(c)
        bc = BoundaryInterrupts()
        for k in range(tb, tb + 8):
            bc.request_after(k)

        pipe = run(prog, bytearray(memory_from_words([0])), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(memory_from_words([0])),
                             initial_regs(), bc)
        self.assertEqual(len(pipe.interrupts), 2)
        self.assertEqual(pipe.regs[SCR], 2)
        self.assertTrue(pipe.halted)
        # 第二次接纳发生在恢复流第一条指令 h+2 退休边界；其下一条
        # （h+3，即 HALT）为保存的恢复点。
        self.assertEqual(pipe.interrupts[1].saved_pc, h + 3)
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)

    def test_handler_with_branch_and_load_use(self):
        # 处理例程内部含分支与 load-use：中断流的流水线时序仍与
        # 顺序参考在架构结果上一致。
        main = [
            Instruction.addi(1, 0, 0),
            Instruction.load(2, 1, 0),
            Instruction.add(3, 2, 2),       # load-use
            Instruction.beq(3, 3, 2),       # 恒真越过下一条 -> HALT
            Instruction.addi(9, 0, 123),    # 被分支冲刷
            Instruction.halt(),
        ]
        handler = [
            Instruction.addi(SCR, SCR, 1),
            Instruction.addi(SCR, SCR, 1),
        ]
        prog, h = build_program(main, handler=handler)
        mem = memory_from_words([7])
        dry = run(prog, bytearray(mem), initial_regs())
        tcycle = retire_cycle_map(dry)[h]
        tb = boundary_index(dry, h)
        ic = InterruptController().request(tcycle)
        bc = BoundaryInterrupts().request_after(tb)
        pipe = run(prog, bytearray(mem), initial_regs(), ic)
        seq = run_sequential(prog, bytearray(mem), initial_regs(), bc)
        self.assertEqual((pipe.regs[2], pipe.regs[3]), (7, 14))
        self.assertEqual(pipe.regs[9], 0)
        self.assertEqual(pipe.regs[SCR], 2)
        self.assertEqual(pipe.regs, seq.regs)
        self.assertEqual(pipe.memory, seq.memory)
        self.assertEqual(
            [(t.pc, t.event) for t in pipe.trace],
            [(t.pc, t.event) for t in seq.trace],
        )


class TestControllerValidation(unittest.TestCase):
    def test_illegal_vector_rejected_before_injection(self):
        with self.assertRaisesRegex(ValueError, "非法中断向量"):
            InterruptController(vector=1)
        with self.assertRaisesRegex(ValueError, "非法中断向量"):
            BoundaryInterrupts(vector=2)
        ic = InterruptController()
        with self.assertRaisesRegex(ValueError, "非法中断向量"):
            ic.request(5, vector=9)
        # 被拒绝的请求不留任何痕迹（无部分变更）
        self.assertEqual(len(ic), 0)
        # 合法向量即固定 VECTOR_PC
        ic.request(5, vector=VECTOR_PC)
        self.assertTrue(ic.has_pulse(5))

    def test_cycle_range_and_type(self):
        ic = InterruptController()
        with self.assertRaises(ValueError):
            ic.request(0)
        with self.assertRaises(ValueError):
            ic.request(MAX_CYCLES + 1)
        with self.assertRaisesRegex(TypeError, "bool"):
            ic.request(True)  # noqa: FBT003

    def test_duplicate_pulses_coalesce_to_one_bit(self):
        ic = InterruptController()
        ic.request(7).request(7).pulse(7)
        self.assertEqual(len(ic), 1)
        self.assertEqual(set(ic.pulses), {7})

    def test_run_rejects_bad_controller_type(self):
        with self.assertRaises(TypeError):
            run([Instruction.halt()], memory_from_words([0]), interrupts={7})
        with self.assertRaises(TypeError):
            run_sequential(
                [Instruction.halt()],
                memory_from_words([0]),
                interrupts=InterruptController(),
            )


if __name__ == "__main__":
    unittest.main()
