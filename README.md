# tinycpu —— 精确异常的五级流水线 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核：实现 IF/ID/EX/MEM/WB 五级
流水线、EX 级操作数前递、load-use 停顿、分支冲刷，以及**精确异常**
（非法访存延迟到 WB 才报告，年轻指令不得提交，老指令正常退休）。
另附一个**独立**的无流水顺序参考解释器，用于交叉验证。

增量能力：**显式 MMIO 地址范围**。两个设备寄存器由本地小状态机模拟
（读寄存器弹 FIFO 队首、写寄存器追加写记录），设备操作只在确定会提交的
WB 退休点执行且恰好一次——错误分支、异常之后的年轻指令都不会触碰设备。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台

如需隔离环境（可选，非必须——本库零依赖）：

```bash
python3 -m venv .venv && source .venv/bin/activate
```

在本目录（含 `tinycpu/`、`tests/`、`demo.py`、`demo_iteration.py` 的目录）执行：

```bash
# 必要测试（53 个用例，含 700 个确定性随机程序的差分测试与 MMIO 提交点测试）
python -m unittest discover -s tests -v

# 固定输入演示：正常结果（含气泡/冲刷）+ 实际触发的异常边界 + 输入拒绝
python demo.py

# MMIO 增量演示：设备读/写提交点、分支冲刷不触设备、设备空读错误
python demo_iteration.py
```

`demo.py` 与 `demo_iteration.py` 均为真实当场计算，无 sleep、无预录结果，
通常在 0.1 秒内完成（远低于 90 秒上限），不连接任何外设、账号或密钥。

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

指令对象通过构造器创建（非法字段在构造时即拒绝）：

```python
from tinycpu import Instruction, memory_from_words
Instruction.add(3, 1, 2)          # ADD r3, r1, r2
Instruction.addi(1, 0, -5)        # ADDI r1, r0, -5
Instruction.load(4, 1, 8)         # LOAD r4, 8(r1)
Instruction.store(2, 1, 0)        # STORE r2, 0(r1)
Instruction.beq(1, 2, 3)          # BEQ r1, r2, +3
Instruction.halt()
```

## 公开接口

### `tinycpu.run(instructions, memory, initial_regs=None) -> RunResult`

五级流水线执行直到 HALT 退休、非法访存在 WB 报告，或周期预算耗尽。

- `instructions`：`Instruction` 序列，1–200 条，最后一条必须是 `HALT`
- `memory`：非空、长度为 4 的倍数的 `bytes/bytearray`
- `initial_regs`：可选，长度 32 的 32 位无符号整数序列，`r0` 必须为 0

返回 `RunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` | 最终 32 个寄存器（tuple） |
| `memory` | 最终（异常时为"异常前"）数据内存 `bytes` |
| `trace` | `RetireRecord` 序列，严格按 WB 退休顺序 |
| `cycles` | `CycleSlots` 序列，每周期 5 个 `Slot`（IF/ID/EX/MEM/WB） |
| `halted` | 是否由 HALT 正常结束 |
| `fault` / `fault_pc` / `fault_cycle` | 异常类型（`Fault.ADDR` / `Fault.DEV`）、指令下标、报告周期 |

`Slot(index, bubble, reason)`：`reason` 取 `None`/`"drain"`（启动排空）/
`"stall"`（load-use 气泡，沿 EX→MEM→WB 传播）/`"hold"`（停顿冻结 IF/ID）/
`"flush"`（分支冲刷）/`"take"`（分支命中的 EX 槽）/`"fault"`（故障丢弃/报告）。

### `tinycpu.run_sequential(instructions, memory, initial_regs=None) -> SeqResult`

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用流水线核心**，
语义独立实现，共享的只有 `isa.py` 里的指令格式与静态校验。用于得到
独立期望值：最终寄存器、异常前内存、逐条架构效果。

## MMIO 设备寄存器（增量）

显式 MMIO 地址范围定义在 `tinycpu/mmio.py`，与数据内存（上限 1 MiB）严格
分离，基址固定在 2 MiB：

| 地址 | 寄存器 | 已提交时的动作 |
|---|---|---|
| `MMIO_READ_ADDR`（0x200000） | 设备读 | 弹出读队列队首并作为 LOAD 值写回；空队列报 `Fault.DEV` |
| `MMIO_WRITE_ADDR`（0x200004） | 设备写 | 把 32 位值追加到设备写记录 |

```python
from tinycpu import Instruction, MMIODevices, MMIO_READ_ADDR, MMIO_WRITE_ADDR, run

dev = MMIODevices(read_values=[41, 7])   # 本地小状态机，不连接外设
prog = [
    Instruction.addi(1, 0, MMIO_READ_ADDR),
    Instruction.load(2, 1, 0),           # 提交时弹出 41 -> r2
    Instruction.halt(),
]
result = run(prog, memory, devices=dev)
dev.pending_reads   # 剩余读队列 (7,)
dev.write_log       # 已提交的设备写记录
```

`run` / `run_sequential` 新增可选关键字参数 `devices`；不传时 MMIO 地址
按普通非法访存处理（`Fault.ADDR`），基线行为完全不变。

提交语义（与"访问设备寄存器的指令不能被错误路径提前执行"对应）：

1. **只在 WB 提交点访问设备，恰好一次**。设备读弹队列、设备写追加记录都
   发生在该指令退休的同一周期；被分支冲刷或异常丢弃的年轻指令物理上
   不可能已触碰设备（与内存 STORE 同一提交点，无需额外提交缓冲）。
2. **MMIO 与普通 store 的顺序**由唯一的 WB 提交点天然保证：退休顺序即
   设备/内存可见顺序，`RetireRecord.mmio`（"read"/"write"）在提交日志中
   标出实际访问点。
3. **设备读值正常写回**：弹出的值在 WB 写入 `mem_wb.load_data`，与内存
   LOAD 走完全相同的写回与前递路径（load-use 停顿照常生效），不存在
   "先预测一个值随后不校正"的窗口。
4. **设备空读错误**：读队列为空时弹队列失败，在该指令的 WB 以
   `Fault.DEV` 精确报告——该读不写回、之后的年轻指令不提交，此前指令
   全部正常退休。

顺序参考解释器实现同一套设备语义，测试与演示逐条对照设备队列、
写记录、寄存器与退休轨迹。

### 错误与异常

| 类型 | 触发条件 |
|---|---|
| `TypeError` | 参数类型错，如 `bool` 冒充 `int`、`float` 传入整数字段 |
| `ValueError` | 数值非法：NaN/Infinity、寄存器号越界、立即数超出 32 位等 |
| `tinycpu.ProgramError`（`ValueError` 子类） | 程序层面非法：不以 HALT 结尾、条数超界、BEQ 静态目标越界、内存非对齐/为空、r0 初值非零 |
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
tinycpu/mmio.py         MMIO 地址范围与设备小状态机（读队列/写记录）
tinycpu/pipeline.py     五级流水线周期内核（被测核心）
tinycpu/sequential.py   独立顺序参考解释器（不依赖 pipeline）
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_equivalence.py  与顺序参考的差分测试（700 个确定性随机程序）
tests/test_mmio.py      MMIO：提交点恰好一次/冲刷安全/空读错误/顺序对照
demo.py                 固定输入演示
demo_iteration.py       MMIO 增量演示（提交日志中的设备实际访问点）
```

## 已知限制

- 只实现题目规定的 7 条指令，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 只有两种异常：数据访存地址错（非对齐或越界，`Fault.ADDR`）与设备空读
  （`Fault.DEV`）；没有中断、系统调用、除零、溢出异常（加减法按模回绕，
  不视为异常）。
- MMIO 设备为进程内小状态机（FIFO 读队列 + 写记录日志），不建模真实外设
  时序、中断或 DMA；MMIO 范围固定为两个 4 字节寄存器。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 周期内核为教学清晰度优先的直接锁存器模拟，未做性能优化。
