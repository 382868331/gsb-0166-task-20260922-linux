"""tinycpu —— 教学用 32 位小五级流水线 CPU 模拟库。

公开接口见 :mod:`tinycpu` 包的 ``__init__``。
"""

from .isa import (
    ADD,
    ADDI,
    BEQ,
    HALT,
    IRET,
    LOAD,
    STORE,
    SUB,
    VECTOR_PC,
    Fault,
    Instruction,
    InterruptStateError,
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
from .interrupt import (
    BoundaryInterrupts,
    InterruptAccept,
    InterruptController,
)
from .pipeline import (
    CycleSlots,
    CycleLimitError,
    RetireRecord,
    RunResult,
    Slot,
    run,
)
from .sequential import SeqTraceEntry, SeqResult, run_sequential

__all__ = [
    "ADD",
    "ADDI",
    "BEQ",
    "HALT",
    "IRET",
    "LOAD",
    "STORE",
    "SUB",
    "VECTOR_PC",
    "Fault",
    "Instruction",
    "InterruptStateError",
    "MASK32",
    "MAX_CYCLES",
    "MAX_INSTRUCTIONS",
    "NUM_REGS",
    "Op",
    "ProgramError",
    "TinyCPUError",
    "memory_from_words",
    "words_from_memory",
    "BoundaryInterrupts",
    "InterruptAccept",
    "InterruptController",
    "CycleSlots",
    "CycleLimitError",
    "RetireRecord",
    "RunResult",
    "Slot",
    "run",
    "SeqTraceEntry",
    "SeqResult",
    "run_sequential",
]
