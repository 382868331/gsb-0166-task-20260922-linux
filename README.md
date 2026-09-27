# tinycpu —— 精确异常的五级流水线 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核：实现 IF/ID/EX/MEM/WB 五级
流水线、EX 级操作数前递、load-use 停顿、分支冲刷，以及**精确异常**
（非法访存延迟到 WB 才报告，年轻指令不得提交，老指令正常退休）。
另支持**可返回的外部中断**：请求按本地模拟时间（周期）到达，只在 WB 提交
后的指令边界被采纳，handler 以 IRET 返回断点继续执行。
另附一个**独立**的无流水顺序参考解释器，用于交叉验证。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台

在本目录（含 `tinycpu/`、`tests/`、`demo.py` 的目录）执行：

```bash
# 必要测试（56 个用例，含 700 个无中断 + 120 个带中断的确定性随机程序差分测试）
python -m unittest discover -s tests -v

# 固定输入演示：正常结果（含气泡/冲刷）+ 实际触发的异常边界 + 输入拒绝
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
| `IRET` | 中断返回，**仅允许**作为中断处理程序（handler）的最后一条 |

指令对象通过构造器创建（非法字段在构造时即拒绝）：

```python
from tinycpu import Instruction, memory_from_words
Instruction.add(3, 1, 2)          # ADD r3, r1, r2
Instruction.addi(1, 0, -5)        # ADDI r1, r0, -5
Instruction.load(4, 1, 8)         # LOAD r4, 8(r1)
Instruction.store(2, 1, 0)        # STORE r2, 0(r1)
Instruction.beq(1, 2, 3)          # BEQ r1, r2, +3
Instruction.halt()
Instruction.iret()                # IRET（仅 handler 末尾）
```

## 公开接口

### `tinycpu.run(instructions, memory, initial_regs=None) -> RunResult`

五级流水线执行直到 HALT 退休、非法访存在 WB 报告，或周期预算耗尽。

- `instructions`：`Instruction` 序列，1–200 条，最后一条必须是 `HALT`
- `memory`：非空、长度为 4 的倍数的 `bytes/bytearray`
- `initial_regs`：可选，长度 32 的 32 位无符号整数序列，`r0` 必须为 0

可选关键字参数（**必须同时提供或同时省略**）：

- `interrupts`：`InterruptRequest` 序列。`InterruptRequest(cycle)` 表示一个
  外部中断请求在本地模拟时间（流水线周期）`cycle` 到达，`cycle ∈ [1, 2000]`。
- `handler`：中断处理程序，`Instruction` 序列，1–200 条，最后一条必须是
  `IRET`，不得含 `HALT`，`IRET` 只能出现一次，`BEQ` 目标必须落在 handler 内。

返回 `RunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` | 最终 32 个寄存器（tuple） |
| `memory` | 最终（异常时为"异常前"）数据内存 `bytes` |
| `trace` | `RetireRecord` 序列，严格按 WB 退休顺序 |
| `cycles` | `CycleSlots` 序列，每周期 5 个 `Slot`（IF/ID/EX/MEM/WB） |
| `halted` | 是否由 HALT 正常结束 |
| `fault` / `fault_pc` / `fault_cycle` | 异常类型（仅 `Fault.ADDR`）、指令下标、报告周期 |
| `fault_in_handler` | 故障指令是否在 handler 空间（此时 `fault_pc` 是 handler 下标） |
| `interrupts_taken` / `interrupts_pending` | 已采纳的中断次数 / 结束时仍未决的请求数 |
| `interrupt_trace` | `InterruptRecord` 序列（take/return 事件，含断点与未决计数） |

`Slot(index, bubble, reason)`：`reason` 取 `None`/`"drain"`（启动排空）/
`"stall"`（load-use 气泡，沿 EX→MEM→WB 传播）/`"hold"`（停顿冻结 IF/ID）/
`"flush"`（分支冲刷）/`"take"`（分支命中的 EX 槽）/`"fault"`（故障丢弃/报告）/
`"irq"`（中断采纳时被冲刷的年轻指令）。

`RetireRecord.space` 标识退休指令所在空间（`"main"`/`"handler"`）。

### `tinycpu.run_sequential(instructions, memory, initial_regs=None) -> SeqResult`

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用流水线核心**，
语义独立实现，共享的只有 `isa.py` 里的指令格式与静态校验。用于得到
独立期望值：最终寄存器、异常前内存、逐条架构效果。

## 可返回外部中断

`run` 与 `run_sequential` 接受同一组 `interrupts`/`handler` 参数，语义对齐：

1. **只在一致程序状态位置进入**。中断采样发生在每个周期 WB 提交之后；
   被采纳时，所有比 WB 年轻的指令（MEM/EX/ID/IF）都尚未提交任何架构效果，
   被整体冲刷（槽位标记 `"irq"`），断点保存为**最老未提交指令**的下标。
   因此不存在"提交了一半"的指令，也不需要提交缓冲。
2. **从正确位置继续，不重复不遗漏**。handler 末尾的 `IRET` 流到 WB 退休时
   恢复断点（提交点唯一在 WB，此刻管线已排空），被冲刷的指令重新取指执行；
   它们此前未提交，故不会重复生效，也不会被跳过。主程序退休序列与
   无中断基线完全一致（测试断言）。
3. **同周期优先级**：WB 故障报告 > HALT 退休 > 中断采纳。分支在 EX 的
   改向是推测性的：若同周期中断被采纳，分支随年轻指令一并冲刷，
   返回后重新执行，效果恰好一次。故障/停机时未服务的请求保持未决。
4. **handler 内不嵌套，请求不静默丢失**。请求是电平语义：handler 执行
   期间（或故障/停机时）到达的请求计入未决计数；`IRET` 退休后若仍有
   未决，立即再次进入 handler（同一周期 take）。运行结束时仍未服务的
   个数在 `interrupts_pending` 中明确报告。
5. **本地模拟时间**。流水线的模拟时间是周期；顺序参考的模拟时间是
   "已退休指令条数"（每退休一条推进 1）。两个时间域的推进速率不同，
   差分测试因此选用与进入点无关的 handler（只写专用寄存器 r31），
   验证最终架构状态一致、主程序退休序列与基线一致。
6. handler 与主程序**共享**寄存器文件和数据内存（无自动现场保存）；
   handler 内同样可能触发 `ADDR` 故障（此时 `fault_in_handler=True`，
   `fault_pc` 为 handler 下标）。

### 错误与异常

| 类型 | 触发条件 |
|---|---|
| `TypeError` | 参数类型错，如 `bool` 冒充 `int`、`float` 传入整数字段 |
| `ValueError` | 数值非法：NaN/Infinity、寄存器号越界、立即数超出 32 位等 |
| `tinycpu.ProgramError`（`ValueError` 子类） | 程序层面非法：不以 HALT 结尾、条数超界、BEQ 静态目标越界、内存非对齐/为空、r0 初值非零 |
| （同上） | 中断层面非法：`interrupts`/`handler` 只给其一、主程序含 IRET、handler 不以 IRET 结尾/含 HALT/BEQ 越界、请求类型错 |
| `tinycpu.CycleLimitError` | 跑满 2000 周期未结束（**预算耗尽**，与"触发故障/证明无解"严格区分） |

所有静态错误在第一条指令执行前抛出，不会产生部分状态；调用方传入的
`bytearray` 内存不会在失败路径上被修改。

### 输入上限（接口边界）

- 程序：1–200 条指令；BEQ 立即数为 32 位有符号数，但目标必须落在程序内（静态校验）
- 周期：2000（超出抛 `CycleLimitError`；正常程序远到不了：最坏每条指令
  load-use 停顿，200 条约 403 周期）
- 数据内存：1 B–1 MiB（`MAX_MEMORY_BYTES`），长度须为 4 的倍数
- 立即数/寄存器初值：32 位范围；寄存器号 0–31
- 中断：请求周期 1–2000，请求个数 ≤ 2000；handler 1–200 条

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
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_equivalence.py  与顺序参考的差分测试（700 个确定性随机程序）
tests/test_interrupt.py  中断：进入/返回、优先级、未决保留、差分（120 个随机程序）
demo.py                 固定输入演示
```

## 已知限制

- 只实现题目规定的 7 条指令，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 只有一种异常：数据访存地址错（非对齐或越界）；没有中断、系统调用、
  除零、溢出异常（加减法按模回绕，不视为异常）。中断为单一外部线，
  无优先级向量/屏蔽寄存器；handler 无自动现场保存（与主程序共享寄存器/内存）。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 周期内核为教学清晰度优先的直接锁存器模拟，未做性能优化。
