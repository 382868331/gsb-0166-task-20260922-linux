# tinycpu —— 精确异常与显式中断的五级流水线 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核：实现 IF/ID/EX/MEM/WB 五级
流水线、EX 级操作数前递、load-use 停顿、分支冲刷、**精确异常**
（非法访存延迟到 WB 才报告，年轻指令不得提交，老指令正常退休），以及
**显式中断请求与 IRET**（退休边界接纳、保存下一条未退休指令的 PC、
冲刷年轻指令、屏蔽嵌套直到 IRET）。另附一个**独立**的无流水顺序参考
解释器，用于在相同退休边界注入事件做交叉验证。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台

在本目录（含 `tinycpu/`、`tests/`、`demo.py` 的目录）执行：

```bash
# 必要测试（55 个用例，含 700 个确定性随机程序的差分测试与中断差分/模糊测试）
python -m unittest discover -s tests -v

# 固定输入演示：正常结果（含气泡/冲刷）+ 实际触发的异常边界 + 输入拒绝
#              + 中断正常路径 + 异常与中断同拍边界
python demo.py
```

`demo.py` 为真实当场计算，无 sleep、无预录结果，通常 0.1 秒内完成。

## ISA 子集

32 个 32 位寄存器，`r0` 恒为零（写 r0 被架构丢弃）。数据内存为小端字节，
按 4 字节字对齐访问。所有算术按 32 位二进制定点回绕（寄存器中存无符号位模式）。

| 指令 | 语义 |
|---|---|
| `ADD rd, rs1, rs2` | `rd = (rs1 + rs2) mod 2^32` |
| `SUB rd, rs1, rs2` | `rd = (rs1 - rs2) mod 2^32` |
| `ADDI rd, rs1, imm` | `imm` 为 32 位有符号立即数 |
| `LOAD rd, imm(rs1)` | 有效地址 `rs1 + imm`，按 4 对齐且在内存内；读 32 位小端字 |
| `STORE rs2, imm(rs1)` | 同上；写 32 位小端字（在 **WB** 级提交） |
| `BEQ rs1, rs2, off` | `off` 以**字（指令）**为单位，目标 = 当前 pc + off |
| `HALT` | 停机，程序必须以 HALT 结尾 |
| `IRET` | 中断返回：恢复接纳时保存的 PC，随后解除中断屏蔽（无中断上下文时为空操作） |

IRET 是新增的唯一一条指令；既有七条指令的编码、字段与语义完全不变。

指令对象通过构造器创建（非法字段在构造时即拒绝）：

```python
from tinycpu import Instruction, memory_from_words
Instruction.add(3, 1, 2)          # ADD r3, r1, r2
Instruction.addi(1, 0, -5)        # ADDI r1, r0, -5
Instruction.load(4, 1, 8)         # LOAD r4, 8(r1)
Instruction.store(2, 1, 0)        # STORE r2, 0(r1)
Instruction.beq(1, 2, 3)          # BEQ r1, r2, +3
Instruction.halt()
Instruction.iret()                # IRET（中断返回）
```

## 公开接口

### `tinycpu.run(instructions, memory, initial_regs=None, *, interrupts=None, interrupt_vector=None) -> RunResult`

五级流水线执行直到 HALT 退休、非法访存在 WB 报告，或周期预算耗尽。

- `instructions`：`Instruction` 序列，1–200 条，最后一条必须是 `HALT`
- `memory`：非空、长度为 4 的倍数的 `bytes/bytearray`
- `initial_regs`：可选，长度 32 的 32 位无符号整数序列，`r0` 必须为 0
- `interrupts` / `interrupt_vector`：可选，**必须同时提供**；启用显式中断
  （见下文"中断与 IRET"）。非法向量在任何周期执行之前拒绝注入

返回 `RunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` | 最终 32 个寄存器（tuple） |
| `memory` | 最终（异常时为"异常前"）数据内存 `bytes` |
| `trace` | `RetireRecord` 序列，严格按 WB 退休顺序 |
| `cycles` | `CycleSlots` 序列，每周期 5 个 `Slot`（IF/ID/EX/MEM/WB） |
| `halted` | 是否由 HALT 正常结束 |
| `fault` / `fault_pc` / `fault_cycle` | 异常类型（仅 `Fault.ADDR`）、指令下标、报告周期 |
| `irq_events` | `IrqEvent` 序列：每次中断接纳的边界、保存 PC、向量 PC、周期 |
| `irq_pending` / `irq_masked` | 运行结束时控制器的 pending 位与屏蔽状态 |

`Slot(index, bubble, reason)`：`reason` 取 `None`/`"drain"`（启动排空）/
`"stall"`（load-use 气泡，沿 EX→MEM→WB 传播）/`"hold"`（停顿冻结 IF/ID）/
`"flush"`（分支冲刷）/`"take"`（分支命中的 EX 槽）/`"fault"`（故障丢弃/报告）/
`"irq"`（中断接纳冲刷的年轻指令）/`"iret"`（IRET 改向的 EX 槽）。

## 中断与 IRET

中断系统由三部分组成：`InterruptController`（离线控制器）、固定向量 PC、
`IRET` 指令。**既有指令编码不变**，未启用中断时行为与原模型逐位一致。

### `tinycpu.InterruptController(pulse_boundaries=())`

离线中断控制器模型：纯软件状态机，**不连接真实 CPU**，由执行器在每个
退休边界步进。

- `request()`：到达一个请求脉冲，把 pending 置 1。pending 只有 **1 位**：
  多个脉冲合并，不计数；未被接纳前一直保持。
- `pulse_boundaries`：构造时给出的退休边界下标（0 起）集合，执行器在第 k
  次退休的边界上自动到达一个脉冲。流水线与顺序参考用**同一个边界编号**
  注入，因此两侧注入点对齐、结果可逐位比较（测试即如此差分）。
- `pending` 属性可随时读取；运行结束后也可从结果的 `irq_pending` 观察。

### 语义契约（流水线与顺序参考完全一致）

1. **接纳点**：只在退休边界评估——当前最老指令在 WB **正常退休之后**。
2. **保存 PC**：接纳时保存"下一条尚未退休指令的 PC"（流水线中最老的
   在役年轻指令：MEM→EX→ID→IF→取指指针），记录进 `IrqEvent.saved_pc`。
3. **冲刷**：全部更年轻指令被冲刷（槽位标记 `"irq"`），PC 转向固定向量。
   被冲刷的 STORE 不可能已写内存（内存写只在 WB 提交点发生）。
4. **屏蔽**：接纳即屏蔽嵌套中断；屏蔽期间的请求仍锁存进 pending（1 位）。
5. **IRET**：在 EX 级恢复保存的 PC（冲刷两条年轻指令，同分支）；屏蔽在
   IRET **退休之后**才解除——IRET 自己的退休边界仍按屏蔽评估，最早下一
   个边界才可再次接纳。无中断上下文的 IRET 是空操作。
6. **异常优先**：同一边界有同步访存异常时，报 `ADDR` 并停机，中断保持
   pending（可从 `irq_pending` 观察到），不接纳。
7. **HALT**：HALT 退休后不再接纳中断；pending 保持。
8. **非法向量**：向量必须是落在程序下标范围内的整数，否则在任何周期
   执行之前抛 `ProgramError`/`TypeError`，拒绝注入，不产生部分状态。

### 处理器（handler）的放置

程序仍须以 `HALT` 结尾（静态契约未变），因此处理器放在程序**中部**，
主程序用一条 `BEQ r0, r0, +k` 跳过它；向量指向处理器第一条指令，
处理器以 `IRET` 结束。`tests/test_interrupt.py` 的 `build()` 即此布局。

### `tinycpu.IrqEvent`

| 字段 | 含义 |
|---|---|
| `boundary` | 第几次退休（0 起）之后接纳，可直接索引退休轨迹 |
| `saved_pc` | 保存的"下一条尚未退休指令"的 PC（IRET 回到这里） |
| `vector_pc` | 转向的固定向量 PC |
| `cycle` | 流水线中的接纳周期；顺序参考无周期概念，为 `None` |

### `tinycpu.run_sequential(instructions, memory, initial_regs=None, *, interrupts=None, interrupt_vector=None) -> SeqResult`

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用流水线核心**，
语义独立实现，共享的只有 `isa.py` 里的指令格式与静态校验、以及
`interrupt.py` 的控制器/事件类型。用于得到独立期望值：最终寄存器、
异常前内存、逐条架构效果、中断事件。中断参数与 `run` 完全同义，
注入按**相同退休边界编号**发生，因此两侧 `irq_events` 逐点可比。

### 错误与异常

| 类型 | 触发条件 |
|---|---|
| `TypeError` | 参数类型错，如 `bool` 冒充 `int`、`float` 传入整数字段 |
| `ValueError` | 数值非法：NaN/Infinity、寄存器号越界、立即数超出 32 位等 |
| `tinycpu.ProgramError`（`ValueError` 子类） | 程序层面非法：不以 HALT 结尾、条数超界、BEQ 静态目标越界、内存非对齐/为空、r0 初值非零、非法中断向量、控制器/向量只给其一 |
| `tinycpu.CycleLimitError` | 跑满 2000 周期未结束（**预算耗尽**，与"触发故障/证明无解"严格区分） |

所有静态错误在第一条指令执行前抛出，不会产生部分状态；调用方传入的
`bytearray` 内存不会在失败路径上被修改。

### 输入上限（接口边界）

- 程序：1–200 条指令；BEQ 立即数为 32 位有符号数，但目标必须落在程序内（静态校验）
- 周期：2000（超出抛 `CycleLimitError`；正常程序远到不了：最坏每条指令
  load-use 停顿，200 条约 403 周期）
- 数据内存：1 B–1 MiB（`MAX_MEMORY_BYTES`），长度须为 4 的倍数
- 立即数/寄存器初值：32 位范围；寄存器号 0–31

## 精确异常与时序设计取舍

1. **提交点唯一在 WB**。寄存器写回与 STORE 内存写都在 WB 执行。非法访存
   在 EX/MEM 已被识别（`bad_addr`），但只置标记、不访问内存，等该指令流到
   WB 才报 `ADDR` 并立即停机。因此故障当周期，更年轻的指令（MEM/EX/ID/IF）
   必然尚未提交——年轻 STORE 物理上不可能已写内存；更老指令已全部退休。
   这就是"精确异常"，不需要额外的提交缓冲。
2. **load-use 停一周期**。EX 中的指令是 LOAD 且 ID 中的指令读其 rd（rd≠0）时，
   冻结 PC 与 IF/ID，向 EX 插入一个 stall 气泡（LOAD 本身在同周期的 EX 正常
   计算地址，下一周期正常进入 MEM）。
3. **EX 前递，最新优先**。取操作数时依次查 EX/MEM 锁存器（最新）、MEM/WB
   锁存器、ID 级捕获的寄存器值；同周期 WB 先写寄存器后 ID 读寄存器。
4. **BEQ 在 EX 决策**。命中时冲刷决策周期 ID、IF 中的两条年轻指令，PC 改向
   目标，目标下周期重新取指。冲刷/停顿气泡作为真实的无效锁存器沿
   EX→MEM→WB 向后传播，在每周期槽位输出中可见。
5. **程序必须以 HALT 结尾且 BEQ 目标静态可查**。这是本子集的刻意收窄：
   顺序执行"掉出程序末尾"没有定义意义，故直接拒绝而非隐式停机。
6. LOAD 读内存在 MEM 级发生；地址非法时不触碰内存（避免越界切片）。
7. 顺序参考是独立实现，期望值不依赖被测核心；"两轮流水线结果相同"式的
   往返一致只在测试中作为补充性结构断言。

## 浮点 / 数值误差口径

本模块**没有任何浮点运算**：指令集只有 32 位二进制定点整数，加减法以
2³² 为模回绕，LOAD/STORE 为精确的小端字节重解释。因此不存在浮点舍入或
容差问题，测试全部为精确相等比较。浮点值仅在输入校验中出现：NaN、
+Infinity、-Infinity 作为数值输入一律拒绝。

## 目录结构

```
tinycpu/__init__.py     公开接口导出
tinycpu/isa.py          指令/常量/校验/内存辅助/错误类型
tinycpu/pipeline.py     五级流水线周期内核（被测核心）
tinycpu/sequential.py   独立顺序参考解释器（不依赖 pipeline）
tinycpu/interrupt.py    离线中断控制器 / IrqEvent / 中断参数校验
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_equivalence.py  与顺序参考的差分测试（700 个确定性随机程序）
tests/test_interrupt.py 中断：接纳/保存PC/冲刷/屏蔽/异常优先/非法向量/差分
demo.py                 固定输入演示
```

## 已知限制

- 只实现题目规定的 7 条指令加 IRET，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 只有一种异常：数据访存地址错（非对齐或越界）；没有系统调用、
  除零、溢出异常（加减法按模回绕，不视为异常）。
- 中断为单一固定向量、1 位 pending（多脉冲合并，不计数）、无优先级/
  无嵌套计数；IRET 无中断上下文时为空操作；控制器是离线软件模型，
  不连接真实 CPU/操作系统。
- 程序仍须以 HALT 结尾，故中断处理器须放在程序中部、由主程序跳过
  （见"中断与 IRET"一节）。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 周期内核为教学清晰度优先的直接锁存器模拟，未做性能优化。
