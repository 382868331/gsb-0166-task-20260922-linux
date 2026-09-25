# tinycpu —— 精确异常与显式外部中断的五级流水线 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核：实现 IF/ID/EX/MEM/WB 五级
流水线、EX 级操作数前递、load-use 停顿、分支冲刷、**精确异常**（非法访存
延迟到 WB 才报告，年轻指令不得提交，老指令正常退休），以及**显式外部中断
与 IRET**（固定向量 PC，在最老指令正常退休后的边界接纳，保存下一条尚未
退休指令的 PC，冲刷年轻指令，屏蔽嵌套直到 IRET）。另附一个**独立**的无
流水顺序参考解释器，用于在相同退休边界交叉验证。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行；中断控制器也是
离线逐拍事件源，不连接真实 CPU、不访问网络或硬件。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台
- 解释器统一用仓库内的 `./.venv/bin/python`

在本目录（含 `tinycpu/`、`tests/`、`demo.py` 的目录）执行：

```bash
# 必要测试（56 个用例，含 700 个确定性随机程序的差分测试与中断差分测试）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示：正常路径、精确异常、中断正常路径、异常/中断同拍、输入拒绝
./.venv/bin/python demo.py
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
| `IRET` | 中断返回：恢复接纳时保存的 PC、冲刷年轻指令并解除中断屏蔽 |

**指令编码体系不变**：IRET 只是新增的操作码枚举成员（无字段），中断向量
是固定常量 `VECTOR_PC = 0`（`tinycpu.VECTOR_PC`），不引入向量寄存器、
状态寄存器指令或新的寻址方式。

指令对象通过构造器创建（非法字段在构造时即拒绝）：

```python
from tinycpu import Instruction, memory_from_words
Instruction.add(3, 1, 2)          # ADD r3, r1, r2
Instruction.addi(1, 0, -5)        # ADDI r1, r0, -5
Instruction.load(4, 1, 8)         # LOAD r4, 8(r1)
Instruction.store(2, 1, 0)        # STORE r2, 0(r1)
Instruction.beq(1, 2, 3)          # BEQ r1, r2, +3
Instruction.halt()                # HALT
Instruction.iret()                # IRET
```

## 外部中断与 IRET 语义

事件源是**离线**对象，测试/调用方逐拍登记脉冲，执行器在退休边界采样：

```python
from tinycpu import InterruptController, BoundaryInterrupts, VECTOR_PC

ic = InterruptController()        # 周期内核用：脉冲按拍号
ic.request(10)                    # 第 10 拍有一个中断脉冲（固定向量）
res = run(prog, mem, regs, ic)

bc = BoundaryInterrupts()         # 顺序参考用：脉冲按"退休边界"
bc.request_after(3)               # 第 4 条指令退休后的边界有脉冲
seq = run_sequential(prog, mem, regs, bc)
```

规则（全部由测试固定）：

1. **向量固定、非法向量先拒绝注入**。向量必须是 `VECTOR_PC`；`request`
   时给出其它向量立即抛 `ValueError`，不留任何已登记脉冲。
2. **在退休边界接纳**。每拍脉冲先并入一位 pending；WB 一条指令**正常
   退休后**才考虑接纳。接纳时：保存"下一条尚未退休指令"的 PC（流水线
   中为 MEM/EX/ID/IF 最老有效锁存器，否则取下一取指 PC；被分支冲刷的
   受害者已无效，故保存点天然是分支目标），冲刷全部年轻指令，PC 改向
   `VECTOR_PC`，置屏蔽位。
3. **异常优先，中断仍待处理**。同一退休边界若 WB 报告同步访存异常
   （ADDR），异常立即停机，脉冲不被消费，结果中 `interrupts_pending=True`。
4. **HALT 退休后不再接纳**；脉冲若恰在 HALT 退休拍到达，只留在 pending
   位（机器已停），不转向量。
5. **屏蔽嵌套直到 IRET**。接纳后到 IRET 退休前为屏蔽态，期间脉冲只置
   pending 一位（**不计数**：同拍多个脉冲、连续多拍脉冲都合并）。
6. **IRET**：恢复保存 PC、冲刷年轻指令，**恢复之后才解除屏蔽**；IRET
   自己的退休边界不接纳（最早在恢复流第一条指令退休边界）。没有活跃
   中断时执行 IRET 抛 `tinycpu.InterruptStateError`。
7. **被冲刷指令无架构效果**：接纳/IRET 冲刷的年轻 STORE 不到 WB，绝不
   写内存；寄存器写同理不发生。
8. **向量为 0 与冷启动共存约定**：向量固定在 pc0，冷启动也从 pc0 起。
   使用中断的程序在 pc0 放一个分发（如 `BEQ r30, r0, +2`：中断进入时
   r30=0 落入 pc2 处理例程；冷启动由调用方预置 r30=1，pc1 无条件分支
   跳过例程），处理例程以 IRET 结尾。`demo.py` 与 `tests/test_interrupts.py`
   是完整示例；不使用中断的程序无需任何改动。

## 公开接口

### `tinycpu.run(instructions, memory, initial_regs=None, interrupts=None) -> RunResult`

五级流水线执行直到 HALT 退休、非法访存在 WB 报告，或周期预算耗尽。

- `instructions`：`Instruction` 序列，1–200 条，最后一条必须是 `HALT`
- `memory`：非空、长度为 4 的倍数的 `bytes/bytearray`
- `initial_regs`：可选，长度 32 的 32 位无符号整数序列，`r0` 必须为 0
- `interrupts`：可选 `InterruptController`；`None` 时与无中断版本完全一致

返回 `RunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` | 最终 32 个寄存器（tuple） |
| `memory` | 最终（异常时为"异常前"）数据内存 `bytes` |
| `trace` | `RetireRecord` 序列，严格按 WB 退休顺序 |
| `cycles` | `CycleSlots` 序列，每周期 5 个 `Slot`（IF/ID/EX/MEM/WB） |
| `halted` | 是否由 HALT 正常结束 |
| `fault` / `fault_pc` / `fault_cycle` | 异常类型（仅 `Fault.ADDR`）、指令下标、报告周期 |
| `interrupts` | `InterruptAccept(cycle, saved_pc, retired_pc)` 序列，按接纳顺序 |
| `interrupts_pending` | 结束时是否还有未消费的 pending 位（异常/HALT 同拍脉冲时为 True） |
| `masked` | 结束时是否处于中断屏蔽态（屏蔽期异常停机时为 True） |

`RetireRecord` 在原字段上新增：`event`（`"INTR"` 接纳 / `"IRET"` 返回 /
`None`）与 `saved_pc`（接纳时保存的恢复点）。接纳事件挂在该边界退休的
那条指令记录上；IRET 事件挂在 IRET 记录上。

`Slot(index, bubble, reason)`：`reason` 取 `None`/`"drain"`（启动排空）/
`"stall"`（load-use 气泡，沿 EX→MEM→WB 传播）/`"hold"`（停顿冻结 IF/ID）/
`"flush"`（分支冲刷）/`"take"`（分支命中的 EX 槽）/`"fault"`（故障丢弃/报告）/
`"intr"`（中断接纳冲刷的年轻指令与传播气泡）/`"iret"`（IRET 冲刷）。

### `tinycpu.run_sequential(instructions, memory, initial_regs=None, interrupts=None) -> SeqResult`

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用流水线
核心**，语义独立实现，共享的只有 `isa.py` 的指令格式与静态校验，中断
逻辑也在此独立重写。`interrupts` 为退休边界脉冲表
（`BoundaryInterrupts`）。用于得到独立期望值：最终寄存器、异常前内存、
逐条架构效果、中断接纳序列与保存 PC。

### 中断事件源（`tinycpu/interrupt.py`）

| 对象 | 方法 | 说明 |
|---|---|---|
| `InterruptController(vector=VECTOR_PC)` | `request(cycle, vector=VECTOR_PC)`（别名 `pulse`）、`has_pulse`、`pulses`、`len` | 周期级脉冲表；同拍重复合并一位 |
| `BoundaryInterrupts(vector=VECTOR_PC)` | `request_after(retired_count)`、`has_pulse`、`boundaries` | 退休边界脉冲表，供顺序参考 |

### 错误与异常

| 类型 | 触发条件 |
|---|---|
| `TypeError` | 参数类型错，如 `bool` 冒充 `int`、`float` 传入整数字段、中断源类型错 |
| `ValueError` | 数值非法：NaN/Infinity、寄存器号越界、立即数超 32 位、**非法中断向量** |
| `tinycpu.ProgramError`（`ValueError` 子类） | 程序层面非法：不以 HALT 结尾、条数超界、BEQ 静态目标越界、内存非对齐/为空、r0 初值非零 |
| `tinycpu.InterruptStateError` | 没有活跃中断（未屏蔽、无保存 PC）时 IRET 退休 |
| `tinycpu.CycleLimitError` | 跑满 2000 周期未结束（**预算耗尽**，与"触发故障/证明无解"严格区分） |

所有静态错误在第一条指令执行前抛出，不会产生部分状态；调用方传入的
`bytearray` 内存不会在失败路径上被修改。非法向量在 `request` 时即拒绝，
脉冲表保持不变。

### 输入上限（接口边界）

- 程序：1–200 条指令；BEQ 立即数为 32 位有符号数，但目标必须落在程序内（静态校验）
- 周期：2000（超出抛 `CycleLimitError`；正常程序远到不了：最坏每条指令
  load-use 停顿，200 条约 403 周期；中断/返回各引入一次 4 拍冲刷）
- 中断脉冲周期：[1, 2000]；同一执行器实例的脉冲去重后最多 2000 个拍号
- 数据内存：1 B–1 MiB（`MAX_MEMORY_BYTES`），长度须为 4 的倍数
- 立即数/寄存器初值：32 位范围；寄存器号 0–31

## 精确异常、中断与时序设计取舍

1. **提交点唯一在 WB**。寄存器写回与 STORE 内存写都在 WB 执行。非法访存
   在 EX/MEM 已被识别（`bad_addr`），但只置标记、不访问内存，等该指令流到
   WB 才报 `ADDR` 并立即停机。因此故障当周期，更年轻的指令（MEM/EX/ID/IF）
   必然尚未提交——年轻 STORE 物理上不可能已写内存；更老指令已全部退休。
   这就是"精确异常"，不需要额外的提交缓冲。
2. **load-use 停一周期**。EX 中的指令是 LOAD 且 ID 中的指令读其 rd（rd≠0）时，
   冻结 PC 与 IF/ID，向 EX 插入一个 stall 气泡（LOAD 本身在同周期的 EX 正常
   计算地址，下一周期正常进入 MEM）。中断在这种停顿周期的边界同样按
   "WB 正常退休"判定，保存点取最老有效锁存器（即被冻结的消费者指令），
   与顺序机的"下一条指令"一致。
3. **EX 前递，最新优先**。取操作数时依次查 EX/MEM 锁存器（最新）、MEM/WB
   锁存器、ID 级捕获的寄存器值；同周期 WB 先写寄存器后 ID 读寄存器。
4. **BEQ 在 EX 决策**。命中时冲刷决策周期 ID、IF 中的两条年轻指令，PC 改向
   目标，目标下周期重新取指。冲刷/停顿气泡作为真实的无效锁存器沿
   EX→MEM→WB 向后传播，在每周期槽位输出中可见。
5. **中断接纳复用 WB 提交点**。接纳只在 WB 一条指令正常退休后的边界发生，
   因此"老指令已提交、年轻指令全部可冲刷"无需额外机制；保存 PC 通过扫描
   MEM→EX→ID→IF 最老有效锁存器得到（冲刷受害者已无效，自动跳过）。
   接纳与 IRET 都把四条年轻锁存器整体作废、PC 改向，冲刷气泡同样沿流水
   向后传播（`intr`/`iret` 标记）。
6. **中断优先级与 pending 一位**。WB 同步异常 > 中断；HALT/IRET 边界不
   接纳。脉冲先并入 pending 再判接纳，保证"异常同拍脉冲不丢失"可从结果
   读出。屏蔽只屏蔽"接纳"，不屏蔽脉冲记录，故多个脉冲只占一位。
7. **程序必须以 HALT 结尾且 BEQ 目标静态可查**。这是本子集的刻意收窄：
   顺序执行"掉出程序末尾"没有定义意义，故直接拒绝而非隐式停机。
8. LOAD 读内存在 MEM 级发生；地址非法时不触碰内存（避免越界切片）。
9. 顺序参考是独立实现，期望值不依赖被测核心；中断的保存/屏蔽/返回语义
   在顺序机上独立重写。两边事件对齐方式是"同一退休边界"（测试先无中断
   干跑取得退休周期↔退休序号映射），而非共享代码。

## 浮点 / 数值误差口径

本模块**没有任何浮点运算**：指令集只有 32 位二进制定点整数，加减法以
2³² 为模回绕，LOAD/STORE 为精确的小端字节重解释。因此不存在浮点舍入或
容差问题，测试全部为精确相等比较。浮点值仅在输入校验中出现：NaN、
+Infinity、-Infinity 作为数值输入一律拒绝。

## 目录结构

```
tinycpu/__init__.py     公开接口导出
tinycpu/isa.py          指令/常量（含 IRET、VECTOR_PC）/校验/内存辅助/错误类型
tinycpu/interrupt.py    离线中断控制器：周期脉冲表 + 退休边界脉冲表
tinycpu/pipeline.py     五级流水线周期内核（被测核心，含中断接纳/IRET）
tinycpu/sequential.py   独立顺序参考解释器（不依赖 pipeline，含中断语义）
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_equivalence.py  与顺序参考的差分测试（700 个确定性随机程序）
tests/test_interrupts.py  中断/IRET：接纳、保存 PC、冲刷、load-use 边界、
                           分支冲刷边界、异常同拍、屏蔽合并、非法向量
demo.py                 固定输入演示
```

## 已知限制

- 只实现题目规定的 8 条指令，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 中断向量固定为 `VECTOR_PC=0`，没有向量寄存器/中断原因寄存器/优先级
  嵌套（只有全局屏蔽一位）；pending 不计数，无法区分脉冲个数与来源。
  向量与冷启动共用 pc0，需要程序自带分发（见上文约定与 demo）。
- 异常只有一种：数据访存地址错（非对齐或越界）；没有系统调用、除零、
  溢出异常（加减法按模回绕，不视为异常）。中断上下文中再触发异常时
  机器停在故障点并保持屏蔽/pending，不自动返回被中断流。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 周期内核为教学清晰度优先的直接锁存器模拟，未做性能优化。
