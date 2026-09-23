# tinycpu —— 精确异常流水线 / 4 项重排序窗口乱序 CPU 模拟模块

一个教学用、可导入的 32 位小 CPU 周期精确内核，提供两个并列核心：

1. **五级顺序流水线**（`tinycpu.run`）：IF/ID/EX/MEM/WB，EX 级操作数前递、
   load-use 停顿、分支冲刷，以及**精确异常**（非法访存延迟到 WB 才报告，
   年轻指令不得提交，老指令正常退休）。
2. **4 项重排序窗口 + 寄存器重命名乱序核心**（`tinycpu.run_out_of_order`）：
   整数算术乱序完成、严格按序退休；分支解析错误时回收窗口中年轻的推测结果；
   LOAD/STORE 保持顺序；非法访存延迟到退休才报告。

另附一个**独立**的无流水顺序参考解释器（`tinycpu.run_sequential`），
与两个核心都不共享执行代码，用于交叉验证（逐退休对照）。

仅使用 Python 3.14.7 标准库，零第三方依赖，完全离线运行。

## 运行环境与命令

- Python：3.14.7（使用 3.10+ 的 `slots=True` dataclass；未使用 3.14 专有特性）
- 依赖：无（`requirements.txt` 仅为占位注释，不需要 `pip install` 任何东西）
- 操作系统：Linux 原生验证，纯标准库代码本身跨平台

在本目录（含 `tinycpu/`、`tests/`、`demo.py` 的目录）执行：

```bash
# 全部测试（62 个用例：700 个顺序流水差分程序、540 个乱序随机差分程序，
# 另含 3.5 万+ 组合的小规模穷举枚举与手算时序用例）
python -m unittest discover -s tests -v

# 固定输入演示：两个核心的正常结果、边界异常与输入拒绝
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

独立顺序参考解释器（`tinycpu/sequential.py`）。**不导入、不调用任何执行
核心**，语义独立实现，共享的只有 `isa.py` 里的指令格式与静态校验。用于得到
独立期望值：最终寄存器、异常前内存、逐条架构效果。

### `tinycpu.run_out_of_order(instructions, memory, initial_regs=None, *, window=4) -> OoORunResult`

4 项重排序窗口（ROB）+ 寄存器重命名（RAT）的乱序核心（`tinycpu/ooo.py`）。
与 `run` 使用**完全相同**的程序/内存/初值静态契约，非法输入同样在第一条
指令执行前抛出。

- `window`：重排序窗口项数（关键字参数），合法范围 1..16，默认
  `REORDER_WINDOW = 4`；取 1 时退化为按序执行。`bool` 显式拒绝。

模型边界（本次迭代的刻意范围）：

- 指令仍按程序序读入窗口；窗口满则取指停顿；**每周期按序退休至多 1 条**。
- 译码期重命名：源操作数记录"等待哪个窗口条目标签"或抓架构寄存器快照；
  目的寄存器记录旧映射。整数算术（ADD/SUB/ADDI）操作数就绪即可发射到
  单个整数运算口（每周期 1 条，选最老的就绪指令，可越过未解析分支推测
  执行）；结果只在窗口标签间流动，架构寄存器只在退休时写入。
- **LOAD/STORE 严格按序**：只有成为窗口队首且操作数就绪才执行；STORE 的
  内存写发生在退休时刻。不做推测 load/store。
- **BEQ 按序解析**（更老的未解析分支优先）；不做预测器，取指沿"不命中"
  路径进行，解析为命中时冲刷窗口中全部更年轻条目（即使已乱序完成，其结果
  随标签回收、RAT 按最年轻→最老还原），PC 改向目标。
- 非法 LOAD/STORE 在执行期识别（不触碰内存），延迟到退休才报告 `ADDR`。

返回 `OoORunResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `regs` / `memory` | 最终寄存器（tuple）/ 最终（异常时为"异常前"）内存 `bytes` |
| `trace` | `RetireRecord` 序列，**严格按退休顺序**（结构同五级流水） |
| `completions` | `CompletionRecord` 序列：乱序**完成**（执行结束）记录，顺序可能与程序序不同；被分支回收的完成 `squashed=True` |
| `window` | `WindowCycle` 序列：每周期末窗口快照（`WindowSlot(tag, pc, op, state)`）、`fetch_pc`、`squashed` 冲刷名单、`redirect` 重定向目标 |
| `halted` / `fault` / `fault_pc` / `fault_cycle` | 同 `RunResult` |

`CompletionRecord(cycle, pc, op, kind, value, addr, bad, taken, squashed)`：
`kind` 为 `"alu"/"load"/"store"/"branch"`；`bad` 标记非法访存。

### 错误与异常

| 类型 | 触发条件 |
|---|---|
| `TypeError` | 参数类型错，如 `bool` 冒充 `int`、`float` 传入整数字段 |
| `ValueError` | 数值非法：NaN/Infinity、寄存器号越界、立即数超出 32 位等 |
| `tinycpu.ProgramError`（`ValueError` 子类） | 程序层面非法：不以 HALT 结尾、条数超界、BEQ 静态目标越界、内存非对齐/为空、r0 初值非零 |
| `tinycpu.CycleLimitError` | 跑满 2000 周期未结束（**预算耗尽**，与"触发故障/证明无解"严格区分），两个执行核心都可能抛 |

所有静态错误在第一条指令执行前抛出，不会产生部分状态；调用方传入的
`bytearray` 内存不会在失败路径上被修改（乱序核心始终在 `validate_program`
返回的内存副本上工作）。`run_out_of_order` 的 `window` 参数非法（非 int、
为 bool、超出 1..16）同样在任何周期之前抛 `TypeError`/`ValueError`。

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

## 乱序核心的时序设计取舍

1. **物理标签即窗口条目标号**，存活标签数 ≤ 窗口大小，不需要独立物理
   寄存器堆；译码时 RAT 把架构寄存器映射到"产生其最新值的标签"，指令的
   源操作数记为（标签，架构快照值），目的寄存器记旧映射以便冲刷还原。
2. **提交点唯一在队首退休**。乱序完成只写窗口条目的结果字段，架构寄存器
   与数据内存只在退休时变更。因此分支冲刷或故障回收年轻结果时，只需丢弃
   窗口后缀条目并还原 RAT——不存在已落架构状态需要撤销。
3. **只有 ADD/SUB/ADDI 允许乱序**。单整数口、每周期 1 条、最老就绪优先；
   它可以越过未解析的 BEQ 与等待中的访存指令执行（演示 3 可见年轻 ADDI
   在 BEQ 解析前先完成），但其推测结果在分支错误时整批回收。
4. **访存顺序靠"队首才可执行"一条规则保证**：年轻 LOAD/STORE 在所有更老
   动态指令退休前不执行；STORE 再额外把内存写推迟到退休。所以不需要
   load/store 队列或内存消歧，就能保证"故障 STORE 之后的年轻 STORE 物理
   上不可能已写内存"。代价是合法的 LOAD 也不能越过老 STORE（本次范围
   明确不做推测 load/store）。
5. **BEQ 不做预测器**：取指沿"不命中"路径填满窗口（年轻算术照常推测
   执行），解析按序（更老的未解析分支优先），命中时冲刷窗口后缀并把
   `fetch_pc` 改向；被误取的 HALT 也会被冲刷，冲刷后恢复取指。
6. **HALT 无执行相位**：读入窗口即标记完成，排到队首时退休停机。
7. `window=1` 是天然的退化模式（无乱序空间），测试用它和 2/3/4/7/8 等
   窗口大小分别对同一批程序做差分：架构结果必须与窗口大小无关。

## 浮点 / 数值误差口径

本模块**没有任何浮点运算**：指令集只有 32 位二进制定点整数，加减法以
2³² 为模回绕，LOAD/STORE 为精确的小端字节重解释。因此不存在浮点舍入或
容差问题，测试全部为精确相等比较。浮点值仅在输入校验中出现：NaN、
+Infinity、-Infinity 作为数值输入一律拒绝。

## 目录结构

```
tinycpu/__init__.py     公开接口导出
tinycpu/isa.py          指令/常量/校验/内存辅助/错误类型
tinycpu/pipeline.py     五级顺序流水线周期内核（原有，未改动语义）
tinycpu/ooo.py          4 项重排序窗口 + 重命名乱序核心（本次新增）
tinycpu/sequential.py   独立顺序参考解释器（不依赖任何执行核心）
tests/test_inputs.py    输入校验与错误语义
tests/test_pipeline.py  顺序流水线时序：停顿/前递/冲刷/精确异常/上限
tests/test_ooo.py       乱序核心：手算时序、重命名/回收、访存按序、
                        window 参数、540 个差分程序与结构不变量
tests/test_equivalence.py  顺序流水与顺序参考的差分测试（700 个随机程序）
demo.py                 固定输入演示（顺序流水 2 段 + 乱序核心 3 段）
```

## 已知限制

- 只实现题目规定的 7 条指令，不是完整 RISC-V/MIPS 行业标准实现，不做此类宣称。
- 无指令内存/取指异常概念；指令以下标数组给出，不做取指地址翻译。
- 只有一种异常：数据访存地址错（非对齐或越界）；没有中断、系统调用、
  除零、溢出异常（加减法按模回绕，不视为异常）。
- BEQ 目标静态校验落在程序内；后向分支死循环由 2000 周期预算截断为
  `CycleLimitError`，这是预算限制而非程序故障（两个核心皆然）。
- 内存上限 1 MiB 为本实现的固定边界（题目未规定数据内存大小）。
- 乱序核心为教学模型：**仅整数算术乱序**；单整数口、单访存口；不做
  推测 load/store（LOAD 不能越过老 STORE）、不做分支预测器、不实现
  异步精确中断；窗口大小上限 16 项。
- 周期数与五级顺序流水线不可直接比较（两者的"一周期"相位定义不同），
  跨内核一致性只按架构结果（寄存器/内存/退休序列/故障位置）核对。
- 周期内核为教学清晰度优先的直接结构模拟，未做性能优化。
