# pipeline_cpu：精确异常的五级流水线小 CPU 模拟库

一个周期精确（cycle-accurate）的教学型 32 位小 CPU 模拟器，实现
**IF / ID / EX / MEM / WB 五级流水线**，带操作数前递、load-use 停顿、
分支冲刷与**精确异常**（precise exception）。同时提供一个完全独立的
无流水顺序解释器作为参照，用于交叉验证。

## 运行环境

- **Python 3.14.7**（仅使用标准库；`from __future__ import annotations`
  等语法在 3.10+ 亦可运行，但以 3.14.7 为验收版本）
- Linux 原生运行，**无需安装任何第三方依赖**，无需网络 / Docker / 账号。

## 准确命令

```bash
# 执行全部必要测试（在仓库根目录）
python -m unittest discover -s tests -v

# 运行固定输入演示（约 0.05 秒，远低于 8 秒预算；真实计算，无 sleep）
python demo.py
```

## 机器模型（公开契约）

- 32 个 32 位寄存器 `r0..r31`，**r0 硬连线为 0**（写入被丢弃）。
- 字可寻址内存：字节地址、**必须 4 字节对齐**，字按**小端序**存放。
  程序（指令数组）位于独立地址空间、从地址 0 连续存放；数据内存由
  调用方给定。
- 算术为 32 位 two's complement **回绕**，不检测溢出。
- 指令集仅 7 条：

| 指令 | 语义 |
|---|---|
| `ADD rd, rs, rt` | `rd = rs + rt`（mod 2^32） |
| `SUB rd, rs, rt` | `rd = rs - rt`（mod 2^32） |
| `ADDI rt, rs, imm` | `rt = rs + imm`；imm 为有符号 16 位 |
| `LOAD rt, rs, imm` | `rt = Mem[rs + imm]` |
| `STORE rs, rt, imm` | `Mem[rs + imm] = rt`（rs 基址，rt 待存数据） |
| `BEQ rs, rt, imm` | 若 `rs == rt` 则 `PC += imm`；imm 为有符号 21 位、4 对齐 |
| `HALT` | 正常停机 |

`Instruction` 字段布局：R 型 `a=rd, b=rs, c=rt`；I 型 `a=rt(目标),
b=rs(源), imm`；STORE `b=rs(base), c=rt(数据), imm`；BEQ `b=rs,
c=rt, imm`。未使用字段必须为 0（否则报输入错误）。

### 接口

```python
from pipeline_cpu import (
    Instruction, Opcode, run_pipeline, run_sequential,
    InvalidProgramError, CycleLimitExceeded,
)

result = run_pipeline(program, memory, max_cycles=2000)
# program : list/tuple[Instruction]，1..200 条
# memory  : bytes/bytearray，长度为 4 的倍数，<= 2^30 字节（内部用副本）
# 返回 SimulationResult：
#   halted, fault(None|Fault), fault_seq, fault_cycle, cycles,
#   retirements: tuple[Retirement]      # 退休轨迹（WB 顺序）
#   trace: tuple[CycleSlots]            # 每周期 IF/ID/EX/MEM/WB 五槽快照
#   final_registers: tuple[int,...](32) # 异常前/停机时寄存器
#   final_memory: bytes                 # 异常前/停机时内存

ref = run_sequential(program, memory)   # 独立顺序解释器，同型输入
```

`CycleSlots` 每槽为 `StageState(seq, stalled, flushed)`，`seq` 为指令
静态下标、气泡为 `None`；`note` 取 `"" / "stall" / "flush" /
"stall+flush" / "halt" / "fault"`。

### 输入上限（接口边界）

| 边界 | 值 |
|---|---|
| 单次程序指令数 | 1..**200** |
| 周期预算 `max_cycles` | 1..**2000**（默认 2000），耗尽抛 `CycleLimitExceeded` |
| 数据内存 | 4 的倍数字节，最大 2^30 字节 |
| 普通立即数 | 有符号 16 位（-32768..32767） |
| 分支偏移 | 有符号 21 位且 4 对齐（-1048576..1048575，步进 4） |
| 寄存器号 | 0..31 |

### 错误语义

- 所有整数参数必须是 `int`：**拒绝 `bool`**（`True/False` 不是合法
  整数），拒绝任何 `float`（因此 **NaN / Infinity 一律拒绝**——本模型
  不存在浮点入参）。
- 非法输入立即抛 `InvalidProgramError`，消息含可定位信息（指令下标、
  操作码、字段名、寄存器号或地址），**绝不静默纠正/截断**。
- 模拟为纯函数：内部使用程序/内存副本，调用方数据不被修改；输入校验
  在任何状态分配前完成，失败不产生部分结果。
- **预算耗尽与停机/无解严格分开**：超过周期上限抛
  `CycleLimitExceeded`，它表示预算不足，不是“程序停机”也不是“证明
  死循环”。

### 精确异常语义

- 非法 LOAD/STORE 在 MEM 阶段发现（越界或未对齐 →
  `Fault("bad_address", addr)`，且**非法 STORE 不写内存**），但
  **到达 WB 才作为故障退休**；故障退休的同一周期，其前方所有年轻
  阶段（MEM/EX/ID/IF）整体冻结丢弃——年轻 STORE 不可能已经写内存。
- 取指出界（顺序跑飞或分支目标越界）产生 `Fault("bad_fetch", addr)`
  随流动到 WB；它也可能在年轻阶段被分支冲刷而取消。
- 故障点的寄存器/内存即“异常前”状态：更老指令全部按序提交，故障
  指令及更年轻指令无任何结构效果。顺序解释器在即将执行故障指令前
  停止，两者快照逐值可比。

## 设计取舍

1. **逆行求值**：周期内严格按 WB→MEM→EX→ID→IF 求值，各阶段读周期
   开始快照。这使“故障/HALT 在 WB 退休即冻结全管”只需一处判断，
   从结构上保证精确异常，无需回滚。
2. **前递优先级**：EX 段操作数依次查 EX/MEM（最新）→ MEM/WB →
   寄存器堆。EX/MEM 中的 LOAD 数据本周期才读取，不可前递；该情形
   由 ID 段 load-use 检测提前挡住（PC 与 IF/ID 冻结一周期、ID/EX
   进气泡）。写 r0 的 LOAD 不触发停顿（r0 恒零）。
3. **分支在 EX 决策**：冲刷当周期 IF/ID 中的两条年轻指令（ID 槽
   标记 `flushed`），IF 直接改取目标；无分支延迟槽。
4. **STORE 在 MEM 写内存**（合法地址才写），早于同周期 WB 的退休；
   因故障退休时 MEM 不执行，年轻 STORE 无从落盘。
5. **顺序参考独立**：`reference.py` 不导入 `core`，只共享 `model`
   的数据类型与校验；测试中的期望值（寄存器、内存、故障地址、轨迹）
   来自它而非被测核心。
6. HALT 排干到 WB 才停机；HALT 之后被顺序取入的年轻指令在其退休
   周期整体丢弃。

## 测试

`tests/` 共 42 个用例：

- `test_validation.py`：bool/float(NaN/Inf)/越界/未对齐/未用字段非零/
  上限/内存不可变/预算耗尽类型区分。
- `test_pipeline.py`：五级时序、稳态吞吐、r0 恒零、EX/MEM 与
  MEM/WB 前递、连续同寄存器依赖取最新值、load-use 恰好停一周期
  （含 STORE/BEQ 场景）、分支冲刷、前向后向循环、非法 LOAD/STORE
  到 WB 才报告、年轻 store 不提交、分支冲刷取消非法写、bad_fetch。
- `test_differential.py`：300 个固定种子（`20260922`）确定性随机
  程序与顺序解释器逐项比对（结局、故障类型/地址、32 寄存器、内存、
  退休值序列），并断言两类结局均被覆盖；另含 200 条上限程序与
  2000 周期边界。

## 已知限制 / 非目标

- 仅实现本题契约的 7 条指令与 32×32 位寄存器子集，**不宣称兼容任何
  行业标准 ISA**（非 MIPS/RISC-V 完整实现）；无特权态、中断、外部
  IO、乘除、移位、符号位判定指令。
- 程序与数据内存分置（程序是 `Instruction` 数组而非可写字），故
  无自修改代码；取指地址按程序下标边界判定。
- 不输出波形文件或流水线图，只提供数据结构形式的逐周期轨迹。
- **浮点误差口径：不适用**。本模拟器是纯整数模型，无任何浮点运算或
  浮点结果；所有算术精确到 32 位回绕语义，不存在容差。
- 随机差分仅用前向分支生成保证终止的程序；后向分支由手写循环用例
  覆盖。无限循环只能靠 `max_cycles` 预算发现（抛
  `CycleLimitExceeded`），模拟器不静态判定程序死活。
