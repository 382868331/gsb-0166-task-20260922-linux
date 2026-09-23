# tinycpu —— 五级流水线 + 4 项 ROB 重命名乱序 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核，包含两个**并存、互不依赖**
的执行核心与一个独立顺序参考：

1. **顺序流水线核心** `run`：IF/ID/EX/MEM/WB 五级，EX 级操作数前递、
   load-use 停顿、分支冲刷，以及**精确异常**（非法访存延迟到 WB 才报告，
   年轻指令不得提交，老指令正常退休）。
2. **乱序核心** `run_ooo`（本次新增）：固定 **4 项重排序窗口（ROB）**与
   **基于标签的寄存器重命名**；整数算术在操作数就绪后**乱序完成**、
   严格**按序退休**；BEQ 静态预测"不命中"，命中即误预测，**回收该分支
   之后所有年轻 ROB 项**（含已经完成的算术结果）并恢复重命名映射；
   **访存严格保序**（LOAD/STORE 不在未解析分支之后投机执行，不做推测
   load/store）。
3. **顺序参考解释器** `run_sequential`：独立无流水实现，用于交叉验证。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台

在本目录（含 `tinycpu/`、`tests/`、`demo.py` 的目录）执行：

```bash
# 全部测试（64 个用例；含旧流水线 700 个 + 乱序核心 600 个确定性随机程序差分）
python -m unittest discover -s tests -v

# 旧五级流水线演示：正常结果（气泡/冲刷）+ 异常边界 + 输入拒绝
python demo.py

# 乱序核心演示：乱序完成/ROB 满停顿/分支回收/访存保序/精确故障/输入拒绝
python demo_ooo.py
```

两个演示均为真实当场计算，无 sleep、无预录结果，通常 0.1 秒内完成。

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
| `fault` / `fault_pc` / `fault_cycle` | 异常类型（仅 `Fault.ADDR`）、指令下标、报告周期 |

`Slot(index, bubble, reason)`：`reason` 取 `None`/`"drain"`（启动排空）/
`"stall"`（load-use 气泡，沿 EX→MEM→WB 传播）/`"hold"`（停顿冻结 IF/ID）/
`"flush"`（分支冲刷）/`"take"`（分支命中的 EX 槽）/`"fault"`（故障丢弃/报告）。

### `tinycpu.run_sequential(instructions, memory, initial_regs=None) -> SeqResult`

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用流水线核心**，
语义独立实现，共享的只有 `isa.py` 里的指令格式与静态校验。用于得到
独立期望值：最终寄存器、异常前内存、逐条架构效果。

### `tinycpu.run_ooo(instructions, memory, initial_regs=None) -> OOORunResult`

在 **4 项 ROB + 寄存器重命名**乱序核心上执行（`tinycpu/ooo.py`），与 `run`
并存、参数与静态校验完全一致（同一 `validate_program`），原有调用不受影响。

每周期固定五个阶段：**退休 → 完成 → 派发(重命名) → 执行(最老就绪，每周期 1 条)
→ 误预测回收**。返回 `OOORunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` / `memory` | 最终 32 寄存器 / 最终（异常时为"异常前"）内存 |
| `trace` | `OOORetireRecord` 序列，**严格按程序序退休** |
| `cycles` | `OOOCycle` 序列，每周期记录派发/执行/完成/退休/回收/停顿事件 |
| `halted` / `fault` / `fault_pc` / `fault_cycle` | 与 `RunResult` 同义 |
| `squashes` | 全部误预测回收事件 `SquashEvent` |

关键事件类型：

- `DispatchEvent(pc, seq, rd, old_tag, src, occupancy)`：派发即重命名。
  `seq` 是全局单调的目标标签；`rd` 是被重命名的架构寄存器（r0/无写回为
  `None`）；`old_tag` 是恢复用旧标签；`src=(t1,t2)` 是两源标签快照
  （`None` 表示读已提交架构寄存器）。
- `OOOCycle.dispatch / execute / complete / retire`：本周期派发的指令、
  开始执行的指令、结果就绪的指令、退休指令（下标，可能含分支导致的空缺）。
  ALU/BEQ/STORE 执行即完成（同周期出现在 `execute` 与 `complete`）；LOAD
  先 `execute`（算地址、开始访存），经 `LOAD_LATENCY=3` 拍才 `complete`。
  `dispatch_stall="rob_full"` 表示 4 项窗口满、前端停发。
- `SquashEvent(cycle, branch_pc, target, squashed, reclaimed_completed)`：
  BEQ 实际命中（静态预测不命中）。`squashed` 是被回收的全部年轻下标，
  `reclaimed_completed` 是其中**已经乱序完成、结果被丢弃**的下标——这是
  "整数算术乱序完成、分支错误回收年轻结果"的直接证据。

行为口径（本迭代边界）：

- **只让整数算术乱序**：ADD/SUB/ADDI（及 BEQ 解析）操作数就绪即可抢先
  完成；结果经标签从已完成项前递，绝不读未提交架构寄存器。
- **按序退休**：寄存器写、STORE 内存写、HALT、ADDR 故障报告都只在 ROB
  头部发生；每条退休写回无条件更新架构寄存器（即便更年轻同寄存器写已
  夺走映射——年轻写若之后被回收或自身故障，架构状态精确停在本条）。
- **访存保序**：LOAD/STORE 启动前要求窗口内无更老的 LOAD/STORE，且更老
  BEQ 全部已解析；故一次至多一条 LOAD 在访存，且访存永不越过未解析分支
  （不做推测 load/store）。STORE 仅在退休时写内存。

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

## 乱序核心（ROB + 重命名）设计取舍

1. **固定 4 项 ROB，按程序序入窗、按序退休**。窗口满时前端停发
   （`dispatch_stall="rob_full"`），不做乱序派发/多发射。
2. **标签重命名**：每条写回指令派发即获得单调标签，RAT 记录架构寄存器
   的当前属主标签，源操作数记录标签快照；WAW/WAR 冒险由标签消除，消费者
   只按标签取已完成值（标签物理值表），r0 不分配标签、恒读 0。
3. **只有整数算术乱序完成**：ALU/BEQ 操作数就绪即可在当拍完成，年轻
   独立 ALU 可早于更老 LOAD 完成；每周期执行选择采用"最老就绪"策略，
   一次只启动 1 条。LOAD 有 `LOAD_LATENCY=3` 拍固定延迟（地址生成+访存）。
4. **分支静态预测不命中、在执行级解析**：实际命中即误预测，回收该分支
   之后全部年轻 ROB 项（连同其已完成标签值），按被回收项逆序恢复 RAT
   旧标签，PC 改向目标、前端重新取指。被回收路径不产生任何架构效果。
5. **访存严格保序、不投机**：LOAD/STORE 启动要求窗口内无更老访存且更老
   BEQ 已解析。因此同址 load-after-store 必然读到老 STORE 写入值；未解析
   分支之后的 STORE 绝不会提前写内存。STORE 地址/数据就绪后仍等退休才写。
6. **精确异常由按序退休自然保证**：非法访存执行期只置标记，等它成为 ROB
   头部才报告 ADDR 并停机；窗口内更年轻指令（含已完成 ALU）整体丢弃，
   更老指令此时已全部退休。退休写回无条件落架构寄存器（见接口说明）。
7. 本迭代**不做**推测 load/store、store-to-load 前递、多发射/多执行单元、
   异步精确中断；这些是明确的范围外项。

## 浮点 / 数值误差口径

本模块**没有任何浮点运算**：指令集只有 32 位二进制定点整数，加减法以
2³² 为模回绕，LOAD/STORE 为精确的小端字节重解释。因此不存在浮点舍入或
容差问题，测试全部为精确相等比较。浮点值仅在输入校验中出现：NaN、
+Infinity、-Infinity 作为数值输入一律拒绝。

## 目录结构

```
tinycpu/__init__.py     公开接口导出
tinycpu/isa.py          指令/常量/校验/内存辅助/错误类型
tinycpu/pipeline.py     五级顺序流水线周期内核
tinycpu/ooo.py          4 项 ROB + 重命名乱序周期内核（本次新增）
tinycpu/sequential.py   独立顺序参考解释器（不依赖 pipeline/ooo）
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_equivalence.py  顺序流水线与顺序参考差分（700 个随机程序）
tests/test_ooo.py       乱序核心：手算时序/回收/保序/穷举/600 个随机差分
demo.py                 顺序流水线固定输入演示
demo_ooo.py             乱序核心固定输入演示（本次新增）
```

## 已知限制

- 只实现题目规定的 7 条指令，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 只有一种异常：数据访存地址错（非对齐或越界）；没有中断、系统调用、
  除零、溢出异常（加减法按模回绕，不视为异常）。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 两个周期内核都为教学清晰度优先的直接模拟，未做性能优化；乱序核心每
  周期只派发/执行 1 条、LOAD 延迟为固定 3 拍，是刻意简化的教学模型。
- 乱序核心不做推测访存、store-to-load 前递、寄存器值物理文件回收压缩
  （退休标签结果保留到运行结束）与异步精确中断——均属本迭代明确范围外。
