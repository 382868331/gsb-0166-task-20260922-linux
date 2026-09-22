"""IF/ID/EX/MEM/WB 五级周期精确流水线（精确异常）。

阶段语义（每个周期内严格按 WB → MEM → EX → ID → IF 逆行求值，
各阶段读取周期开始时的流水寄存器快照、写入下一快照）：

* WB  : 退休指令。写寄存器文件；HALT/故障在此刻才生效。故障或 HALT
        退休的同一周期，其前方所有年轻阶段整体冻结（不执行 MEM/EX/
        ID/IF），因此年轻 STORE 不可能已经写内存——精确异常。
* MEM : LOAD 读内存；STORE 仅在地址合法时写内存，地址非法则打上
        bad_address 故障且不写，随流水到 WB 才报告。
* EX  : ALU 运算 / 计算访存有效地址 / BEQ 决策。操作数经前递网络：
        EX/MEM（最新）优先，其次 MEM/WB，最后读寄存器堆。
* ID  : 译码并读寄存器。在此检测 load-use 冒险：若 EX 段是 LOAD 且
        ID 段指令读其目的寄存器，冻结 PC 与 IF/ID，向 ID/EX 插入一个
        气泡（停恰好一周期）。
* IF  : 按 PC 取指。BEQ 在 EX 决定跳转时冲刷 IF/ID 中的年轻指令，
        丢弃本周期顺序取指，改取分支目标。

取指出界（顺序跑飞或跳转目标越界）产生 bad_fetch 故障标记，随流水
流动；它可能被分支冲刷（年轻、从未退休），也可能到 WB 成为精确故障。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import (
    Fault,
    Instruction,
    InvalidProgramError,
    CycleLimitExceeded,
    Opcode,
    Retirement,
    CycleSlots,
    SimulationResult,
    StageState,
    WORD_MASK,
    validate_program,
    _check_int,
    MAX_CYCLES,
)


@dataclass
class _Tok:
    """流水线中流动的内部令牌（一条指令或一个待处理取指故障）。"""

    seq: Optional[int]
    pc: int
    op: Optional[Opcode]
    inst: Optional[Instruction] = None
    fault: Optional[Fault] = None
    rd: Optional[int] = None          # 写回寄存器号；None 表示不写回
    alu: Optional[int] = None         # EX 算出的 ALU 结果/有效地址
    store_data: Optional[int] = None  # STORE 待写数据
    wb_value: Optional[int] = None    # MEM 阶段后可供 WB 的值


def _mask32(x: int) -> int:
    return x & WORD_MASK


def _sources(inst: Instruction) -> tuple[int, ...]:
    op = inst.opcode
    if op in (Opcode.ADD, Opcode.SUB, Opcode.BEQ):
        return (inst.b, inst.c)
    if op in (Opcode.ADDI, Opcode.LOAD):
        return (inst.b,)
    if op is Opcode.STORE:
        return (inst.b, inst.c)
    return ()  # HALT


def _dest(inst: Instruction) -> Optional[int]:
    if inst.opcode in (Opcode.ADD, Opcode.SUB, Opcode.ADDI, Opcode.LOAD):
        return inst.a
    return None


class _Machine:
    def __init__(self, prog: tuple[Instruction, ...], mem: bytearray) -> None:
        self.prog = prog
        self.mem = mem
        self.regs = [0] * 32
        self.pc = 0
        self.if_id: Optional[_Tok] = None
        self.id_ex: Optional[_Tok] = None
        self.ex_mem: Optional[_Tok] = None
        self.mem_wb: Optional[_Tok] = None
        self.trace: list[CycleSlots] = []
        self.retirements: list[Retirement] = []
        self.fault: Optional[Fault] = None
        self.fault_seq: Optional[int] = None
        self.fault_cycle: Optional[int] = None
        self.halted = False

    # -- 工具 --------------------------------------------------------------

    def _fetch(self, addr: int) -> _Tok:
        n = len(self.prog)
        if addr % 4 == 0 and 0 <= addr < 4 * n:
            seq = addr // 4
            return _Tok(seq=seq, pc=addr, op=self.prog[seq].opcode,
                        inst=self.prog[seq])
        return _Tok(seq=None, pc=addr, op=None,
                    fault=Fault("bad_fetch", _mask32(addr)))

    def _addr_valid(self, addr: int) -> bool:
        return addr % 4 == 0 and 0 <= addr <= len(self.mem) - 4

    def _forward(self, rn: int, ex_mem: Optional[_Tok],
                 mem_wb: Optional[_Tok]) -> int:
        """EX 段操作数前递：EX/MEM 最新优先，其次 MEM/WB，最后寄存器堆。"""
        if rn == 0:
            return 0
        if ex_mem is not None and ex_mem.rd == rn and ex_mem.fault is None:
            # MEM 段指令最新。LOAD 的数据本周期才在读数，不能前递——
            # 该情形已在 ID 段被 load-use 检测挡下（插入气泡）。
            if ex_mem.op is not Opcode.LOAD:
                return _mask32(ex_mem.alu if ex_mem.alu is not None else 0)
        if mem_wb is not None and mem_wb.rd == rn and mem_wb.fault is None:
            val = mem_wb.wb_value
            return _mask32(val if val is not None else 0)
        return self.regs[rn]

    def _retire(self, tok: _Tok, cycle: int) -> None:
        if tok.fault is not None:
            self.fault = tok.fault
            self.fault_seq = tok.seq
            self.fault_cycle = cycle
            self.retirements.append(Retirement(
                seq=tok.seq,
                opcode=tok.op,  # bad_fetch 无对应指令，为 None
                cycle=cycle, rd=None, value=None, fault=tok.fault))
            return
        assert tok.op is not None
        if tok.op is Opcode.HALT:
            self.halted = True
            self.retirements.append(Retirement(
                seq=tok.seq, opcode=tok.op, cycle=cycle, rd=None,
                value=None, halted=True))
            return
        rd = tok.rd
        if rd is not None and rd != 0:
            value = _mask32(tok.wb_value if tok.wb_value is not None else 0)
            self.regs[rd] = value
        else:
            value = None  # 不写回（STORE/BEQ）或目标 r0（恒零，丢弃）
        self.retirements.append(Retirement(
            seq=tok.seq, opcode=tok.op, cycle=cycle, rd=rd, value=value))

    @staticmethod
    def _stage(tok: Optional[_Tok], stalled: bool = False,
               flushed: bool = False) -> StageState:
        return StageState(None if tok is None else tok.seq, stalled, flushed)

    # -- 单周期 ------------------------------------------------------------

    def step(self, cycle: int) -> bool:
        """推进一个周期。返回 False 表示机器终止（HALT/故障已退休）。"""
        # 周期开始时占据各槽的指令（本周期“正在”该阶段的指令）。
        wb_tok = self.mem_wb
        mem_tok = self.ex_mem
        ex_tok = self.id_ex
        id_tok = self.if_id

        terminate = wb_tok is not None and (
            wb_tok.fault is not None or wb_tok.op is Opcode.HALT
        )

        # ---------------- WB：最先求值（退休点） ----------------
        if wb_tok is not None:
            self._retire(wb_tok, cycle)

        if terminate:
            # 老指令已提交；本周期前方年轻指令全部冻结并随终止丢弃，
            # 尤其年轻 STORE 不允许进入 MEM——精确异常。
            note = "fault" if wb_tok.fault is not None else "halt"  # type: ignore[union-attr]
            self.trace.append(CycleSlots(
                cycle=cycle,
                if_stage=StageState(None),
                id_stage=self._stage(id_tok),
                ex_stage=self._stage(ex_tok),
                mem_stage=self._stage(mem_tok),
                wb_stage=self._stage(wb_tok),
                note=note,
            ))
            self.if_id = self.id_ex = self.ex_mem = self.mem_wb = None
            return False

        # ---------------- MEM ----------------
        next_mem_wb: Optional[_Tok] = None
        if mem_tok is not None:
            t = mem_tok
            if t.fault is not None:
                next_mem_wb = t  # 取指故障继续下传
            elif t.op is Opcode.LOAD:
                addr = _mask32(t.alu if t.alu is not None else 0)
                if self._addr_valid(addr):
                    t.wb_value = int.from_bytes(
                        self.mem[addr:addr + 4], "little")
                else:
                    t.fault = Fault("bad_address", addr)
                next_mem_wb = t
            elif t.op is Opcode.STORE:
                addr = _mask32(t.alu if t.alu is not None else 0)
                if self._addr_valid(addr):
                    data = _mask32(
                        t.store_data if t.store_data is not None else 0)
                    self.mem[addr:addr + 4] = data.to_bytes(4, "little")
                else:
                    # 关键：非法 STORE 不写内存，标记后到 WB 才报告。
                    t.fault = Fault("bad_address", addr)
                next_mem_wb = t
            else:
                t.wb_value = t.alu
                next_mem_wb = t

        # ---------------- EX ----------------
        next_ex_mem: Optional[_Tok] = None
        branch_flush = False
        branch_target = 0
        if ex_tok is not None:
            t = ex_tok
            if t.fault is not None:
                next_ex_mem = t
            else:
                inst = t.inst
                assert inst is not None
                t.rd = _dest(inst)
                op = inst.opcode
                if op in (Opcode.ADD, Opcode.SUB):
                    a = self._forward(inst.b, mem_tok, wb_tok)
                    b = self._forward(inst.c, mem_tok, wb_tok)
                    t.alu = _mask32(a + b if op is Opcode.ADD else a - b)
                elif op is Opcode.ADDI:
                    t.alu = _mask32(self._forward(inst.b, mem_tok, wb_tok)
                                    + inst.imm)
                elif op is Opcode.LOAD:
                    t.alu = _mask32(
                        self._forward(inst.b, mem_tok, wb_tok) + inst.imm)
                elif op is Opcode.STORE:
                    t.store_data = self._forward(inst.c, mem_tok, wb_tok)
                    t.alu = _mask32(
                        self._forward(inst.b, mem_tok, wb_tok) + inst.imm)
                elif op is Opcode.BEQ:
                    a = self._forward(inst.b, mem_tok, wb_tok)
                    b = self._forward(inst.c, mem_tok, wb_tok)
                    if a == b:
                        branch_flush = True
                        branch_target = _mask32(t.pc + inst.imm)
                # HALT：无 ALU 动作
                next_ex_mem = t

        # ---------------- ID：load-use 冒险检测 ----------------
        stall = (
            ex_tok is not None
            and ex_tok.inst is not None
            and ex_tok.inst.opcode is Opcode.LOAD
            and id_tok is not None
            and id_tok.inst is not None
            and _dest(ex_tok.inst) not in (None, 0)
            and _dest(ex_tok.inst) in _sources(id_tok.inst)
        )

        # ---------------- IF + 流水寄存器更新 ----------------
        notes: list[str] = []
        if stall:
            # PC 与 IF/ID 冻结；ID/EX 插入气泡，EX 段的 LOAD 继续前进。
            next_id_ex: Optional[_Tok] = None
            next_if_id = id_tok
            fetch_tok = self._fetch(self.pc)  # 同址重取，仅用于轨迹显示
            if_stage = self._stage(fetch_tok, stalled=True)
            id_stage = self._stage(id_tok, stalled=True)
            notes.append("stall")
        elif branch_flush:
            # 冲刷 IF/ID 中的年轻顺序指令；丢弃本周期顺序取指，IF 改取
            # 分支目标，ID/EX 进气泡。
            fetch_tok = self._fetch(branch_target)
            self.pc = _mask32(branch_target + 4)
            next_if_id = fetch_tok
            next_id_ex = None
            if_stage = self._stage(fetch_tok)
            id_stage = self._stage(id_tok, flushed=True)
            notes.append("flush")
        else:
            # 正常：ID 透传（开管首周期为 None=气泡），IF 在当前 PC
            # 取指后 PC += 4。
            next_id_ex = id_tok
            fetch_tok = self._fetch(self.pc)
            self.pc = _mask32(self.pc + 4)
            next_if_id = fetch_tok
            if_stage = self._stage(fetch_tok)
            id_stage = self._stage(id_tok)

        self.mem_wb = next_mem_wb
        self.ex_mem = next_ex_mem
        self.id_ex = next_id_ex
        self.if_id = next_if_id

        self.trace.append(CycleSlots(
            cycle=cycle,
            if_stage=if_stage,
            id_stage=id_stage,
            ex_stage=self._stage(ex_tok),
            mem_stage=self._stage(mem_tok),
            wb_stage=self._stage(wb_tok),
            note="+".join(notes),
        ))
        return True

    def run(self, max_cycles: int) -> SimulationResult:
        # 开管时全部流水寄存器为空，pc=0；cycle 1 的 IF 即取回 I0。
        cycle = 0
        while True:
            cycle += 1
            if cycle > max_cycles:
                raise CycleLimitExceeded(
                    f"流水线在 {max_cycles} 周期内未退休 HALT 或故障")
            alive = self.step(cycle)
            if not alive:
                break
            if all(x is None for x in
                   (self.if_id, self.id_ex, self.ex_mem, self.mem_wb)):
                # 防御性分支：无 HALT 且取指未故障的排空不应发生。
                raise CycleLimitExceeded("流水线排空但未遇到 HALT/故障")
        return SimulationResult(
            halted=self.halted,
            fault=self.fault,
            fault_seq=self.fault_seq,
            fault_cycle=self.fault_cycle,
            cycles=cycle,
            retirements=tuple(self.retirements),
            trace=tuple(self.trace),
            final_registers=tuple(self.regs),
            final_memory=bytes(self.mem),
            retired_count=len(self.retirements),
        )


def run_pipeline(
    program: object,
    memory: object,
    max_cycles: int = MAX_CYCLES,
) -> SimulationResult:
    """对给定程序与对齐内存运行五级流水线，返回完整轨迹结果。

    参数
    ----
    program:
        ``list/tuple[Instruction]``，长度 1..200。
    memory:
        ``bytes/bytearray``，长度为 4 的倍数的数据内存（小端字）。
        函数内部使用副本，不修改调用方数据。
    max_cycles:
        周期预算，默认 2000；超出抛 :class:`CycleLimitExceeded`。

    返回
    ----
    :class:`SimulationResult`
        含退休轨迹、每周期五槽快照、最终寄存器与异常前/停机内存。

    抛出
    ----
    InvalidProgramError
        任何输入不满足契约（含 bool 冒充整数、float/NaN/Infinity 等）。
    CycleLimitExceeded
        周期预算耗尽（区别于程序正常停机或无解证明）。
    """
    mc = _check_int(max_cycles, "max_cycles")
    if not 1 <= mc <= MAX_CYCLES:
        raise InvalidProgramError(
            f"max_cycles 必须在 1..{MAX_CYCLES} 内，得到 {mc}")
    prog, mem = validate_program(program, memory)
    return _Machine(prog, mem).run(mc)
