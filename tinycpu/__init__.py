"""tinycpu —— 教学用 32 位小五级流水线 CPU 模拟库。

公开接口见 :mod:`tinycpu` 包的 ``__init__``。
"""

from .isa import (
    ADD,
    ADDI,
    BEQ,
    HALT,
    LOAD,
    STORE,
    SUB,
    Fault,
    Instruction,
    MASK32,
    MAX_CYCLES,
    MAX_INSTRUCTIONS,
    NUM_REGS,
    Op,
    ProgramError,
    TinyCPUError,
    memory_from_words,
    words_from_memory,
)
from .pipeline import (
    BranchPredictor,
    CycleSlots,
    CycleLimitError,
    RetireRecord,
    RunResult,
    Slot,
    run,
)
from .sequential import SeqTraceEntry, run_sequential

__all__ = [
    "ADD",
    "ADDI",
    "BEQ",
    "HALT",
    "LOAD",
    "STORE",
    "SUB",
    "Fault",
    "Instruction",
    "MASK32",
    "MAX_CYCLES",
    "MAX_INSTRUCTIONS",
    "NUM_REGS",
    "Op",
    "ProgramError",
    "TinyCPUError",
    "memory_from_words",
    "words_from_memory",
    "BranchPredictor",
    "CycleSlots",
    "CycleLimitError",
    "RetireRecord",
    "RunResult",
    "Slot",
    "run",
    "SeqTraceEntry",
    "run_sequential",
]
