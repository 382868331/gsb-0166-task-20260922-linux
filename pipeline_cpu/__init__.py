"""pipeline_cpu: 精确异常的五级流水线小 CPU 模拟库。

公开模块:
    model     -- 数据结构、接口契约、错误类型与上限常量
    core      -- IF/ID/EX/MEM/WB 五级周期精确流水线
    reference -- 无流水顺序解释器(独立参考,不依赖 core)
"""

from .model import (
    Opcode,
    Instruction,
    Fault,
    Retirement,
    StageState,
    CycleSlots,
    SimulationResult,
    ReferenceResult,
    InvalidProgramError,
    CycleLimitExceeded,
    MAX_PROGRAM_INSTRUCTIONS,
    MAX_CYCLES,
    MAX_MEMORY_WORDS,
    REGISTER_COUNT,
    WORD_MASK,
)
from .core import run_pipeline
from .reference import run_sequential

__all__ = [
    "Opcode",
    "Instruction",
    "Fault",
    "Retirement",
    "StageState",
    "CycleSlots",
    "SimulationResult",
    "ReferenceResult",
    "InvalidProgramError",
    "CycleLimitExceeded",
    "MAX_PROGRAM_INSTRUCTIONS",
    "MAX_CYCLES",
    "MAX_MEMORY_WORDS",
    "REGISTER_COUNT",
    "WORD_MASK",
    "run_pipeline",
    "run_sequential",
]
