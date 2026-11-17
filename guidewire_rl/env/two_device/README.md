# 两器械任务的环境零件（导丝 + 钩形导管）

这个文件夹放两器械任务在环境这一边的全部零件：导丝 + lab 的钩形导管，从降主动脉的插入点到右颈内。

- 现在的配置：[`configs/two_devices_ica.yaml`](../../../configs/two_devices_ica.yaml) 和继承它的五份。
- **这里只讲有哪些零件、怎么接、怎么测。** 为什么这样设计、试过哪些不行，在 [`docs/design.md`](../../../docs/design.md)。
- 每个文件开头的说明是那个零件最细的描述（比如裁判的五段、两档、事件都写在 `phases.py` 开头），这里不重复。
- 第一轮（2026-10-31，目标只到右颈总）的观测、奖励、安全壳和当时量的数，在 [`docs/archive/two_device_round1_env.md`](../../../docs/archive/two_device_round1_env.md)，那是这份 README 2026-11-14 以前的原文。

## 零件一览

“配置里的名字”就是 YAML 里 `type:` 后面写的。最后一列：**用** = 现在的配置在用；**第一轮** = 只有 `two_devices_round1.yaml` 用，留着没删。

| 文件 | 零件（配置里的名字） | 管什么 | 现在 |
| --- | --- | --- | --- |
| `route.py` | `Route`（内部用，不进配置） | 路线和尺子：沿血管走多远、进某根血管多深、离目标还有多少路 | 用 |
| `phases.py` | `PhaseDetector`（`env.monitors`）、`PhaseInfo`（`env.info`） | 裁判：五段、两档、事件、硬失败、进度比例、地标格 | 用 |
| `observations.py` | `InstrumentTip2D`、`InstrumentShape2D`、`TaskTarget2D`、`JunctionMap2D`、`TierFlag`、`InsertionDiff`、`InsertionLengthsNormalized`、`FrameHistory`；包装 `TipNoise2D`、`Normalize2DTask`；`TaskBox2D`（`env.monitors`） | 策略看到的东西；画面坐标归一用的框 | 用 |
| `observations.py` | `PhaseOneHot`、`ScriptSuggestion` | 段号、脚本的建议动作 | 第一轮 |
| `reward_formula.py` | `TwoDeviceFormula`（`env.reward_formula`） | 奖励：一份公式，环境实时算一遍，回放抽样时按存下的原料再算一遍 | 用 |
| `rewards.py` | `ZeroReward` | 占住 stEVE 的奖励槽（真正的分由上一行算） | 用 |
| `rewards.py` | `LeadingTipProgress`、`CatheterToF`、`HookPrepPotential`、`PhasePotential`、`WindowPenalty`、`EventPenalty`、`ShieldPenalty`、`StepPenalty`、`OutcomeReward` | 逐项相加的九项奖励 | 第一轮 |
| `terminal.py` | `PlatformReached`、`HardFailure` | 真结束（不往后估价值）：第二档成功、出血管 | 用 |
| `truncation.py` | `TipForceBlowup`、`ForceBlowupInfo`（`env.info`） | 接触力爆掉时截断这一局，当仿真故障 | 用 |
| `truncation.py` | `Stalled`、`StallInfo`（`env.info`） | 两根器械的送入长度在 `window_steps` 步里净变化都不到 `min_mm` 就截断这一局（旋转不算在动）。默认不生效：`env.end_when_stalled()` 只对下一局开（考试用，`evaluation.end_when_stalled`），`always: true` 每局都开 | 只在考试里用 |
| `truncation.py` | `PhaseCutoff` | 到了训练器给的段号就停 | 第一轮 |
| `wall.py` | `WallDistance`（`env.monitors`）、`WallPenetration`（`env.labels`）、`WallBreach`（`env.truncation`）、`WallInfo`（`env.info`） | 拿碰撞网格量器械离血管壁多远；中心线出了网格就截断，当仿真故障 | 用 |
| `actions.py` | `TwoDeviceLimits`（`env.action`）、`ShieldInfo`（`env.info`） | 动作映射：机器人模式层 → 速度上限 → 送入长度限位；报执行了什么 | 用 |
| `actions.py` | `TwoDeviceGuarded`、`NominalAdvisor` | 第一轮的六道安全壳；脚本建议的通道 | 第一轮 |
| `labels.py` | `InstrumentTipForce`、`CatheterTipForce`、`TipClearance`、`SubstepMotion` | 训练答案（不进观测）：两头的受力、离壁距离、一个小步里节点最多挪多少 | 用 |
| `targets.py` | `RouteWindowTarget` | 目标点：沿路线的一段里抽 | 用 |
| `start.py` | `WireFirst`（`env.start`） | 开局：每局先只送导丝到 40 mm | 用 |
| `snapshot.py` | `FromSnapshot`（`env.start`）、`EpisodeSnapshots`；函数 `capture`、`save`、`load`、`restore_simulation` | 仿真存档：存、恢复、训练局沿途存 | 起点库的两份配置用 |
| `prefix.py` | `run_prefix`（函数，训练器在环境进程里调） | 手写推法每局现场走一段再交给策略 | “两根拐棍”用 |
| `shadow.py` | `ShadowExpert`（`env.shadow_expert`） | 手写推法在旁边每步算出“换我会怎么做”，不执行，当模仿的答案 | “老师旁观”用 |
| `script.py` | `StagedController`（不进 `type:`，由上面两行和录示范的脚本用） | 手写推法 | 用 |
| `script_round1.py` | `StagedController` 的第一轮版 | `script.py` 在 2026-11-06 重调之前的原文，逐字留着做对照，有测试锁住 | 对照 |
| `../instruments/hook.py` | `HookCatheter` | lab 的钩形导管 | 用 |

配置里另外几个名字不在这个文件夹：`Rotations`、`LastAction`、`MaxSteps`、`SimError` 等来自 stEVE 或框架。

## 现在的配置怎么接这些零件

- **裁判**：配置里叫 `phase_detector`，每步紧跟仿真走，在观测和奖励之前；别的零件按这个名字拿到同一个实例。它读三维真值，只给奖励、真结束、截断、标签、存档、手写推法用，不进观测（只有“已拿第一档”一位例外）。
- **奖励**：stEVE 的奖励槽放 `ZeroReward`；环境在标签算完以后调 `reward_formula` 算本步的分，原料放进 `info["reward_materials"]`。原料 15 个数（`reward_formula.py` 的 `FIELDS`）：进度、跟进、摆钩点尺子的两个备用量、两头受力、进错血管、第一档、成功、出血管、数值崩、领先、“只算新进展”的两个备用量、动作变化。
- **观测**（120 个数，顺序见配置）：两个头、两段形状、导丝目标、导管目标 F（`TaskTarget2D(point: F)`）、路口地图（`mode: all`，路线上六个路口一次全给；过了目标的路口位置照列、方向置 0）、送入长度、长度差、转角、最近三步执行的动作、第一档那一位、前两帧。所有画面坐标按整棵血管树的框归一（`TaskBox2D`，`region: tree`）。
- **动作**（`TwoDeviceLimits`）：机器人模式层（默认直通，`mode_map` 换）→ 速度上限 → 送入长度限位。执行了什么记在 `info["executed_action"]`，观测里的最近三步动作读它。
- **一局**：`WireFirst` 开局；步数上限从插入点算，前缀走的步也算在内；数值崩、穿墙、力爆掉都是截断，训练器把那一步丢掉。
- **起点课程**（训练器那边，`train/curriculum.py`）：按裁判报的进度比例 `progress_ratio` 分十格，或按地标格 `landmark_grid`。三种供法：`prefix`（`run_prefix` 现场走）、`self` 和 `demos`（从存档恢复，`env.reset(options={"snapshot": 路径})`）。
- **老师旁观**：`env.shadow_expert` 打开后，每步动作执行之前，手写推法按仿真真值算出答案，放进 `info["expert_action"]`；另有 `expert_stage`、`expert_substage`、`expert_off`（这一步没有答案的原因）、`expert_events`。
- **目标**：`RouteWindowTarget(past_fork: v8, …)`，在颈内外分叉往上的一段里抽；候选点按奇偶分成训练用和考试用；`jitter_mm`（默认 0）让训练用的点每局沿路线前后随机挪这么多，考试用的点不挪；`sampling: uniform`（默认 `grid`）让训练用的点在整段里任取。
- **备用、默认关的**：摆钩点尺子（`arch_ruler`；每个头到过摆钩点或进了头臂干以后换回普通尺子）；结束时按领先打折的软要求（`lead_soft_min_mm`）。

配置里怎么写每个键、每个键什么意思，见 [`docs/training_framework.md`](../../../docs/training_framework.md)。

## 尺子怎么量

- 血管树按中心线和路口连成一张图，距离都沿图量。
- 从主动脉分出去的血管，中心线是从主动脉中间起的，头 19–21 mm 在主动脉腔里。它的“口”取中心线穿出主动脉管壁的地方（头臂干口在中心线上 19.7 mm 处）。腔里那一段算 0 长，头端越过口时尺子不跳。从细血管分出去的从路口算起：右颈总从头臂干末端算。
- 一个点属于哪根血管：落在哪根中心线管子里最深就算哪根。管子半径量到模型内壁，和内壁的差：中位 0.2 mm，路线附近最多 3.7 mm。
- 深度 `depth(点, 血管)`：从这根血管的口量起，进去为正，没进为负（负值是回到口还要走的路）。
- 关键位置（路线弧长，从主动脉起点量，mm）：插入点 1.5，头臂干在主动脉上的路口 177.9，右颈总开口 217.9。右颈总开口过去 63.8 mm 有一个向下的侧支，过去 90.9 mm 是颈内外分叉。

## 五段和两档

裁判把一局分成五段：过弓、摆钩、选口、出管、跟进。变段的条件、两档的条件、事件的门槛，都写在 `phases.py` 开头的说明里，门槛的数在配置的 `thresholds`。

五段现在只进统计（`info` 的 `phase`、`max_phase`），不进奖励、不进观测。两处还在用它：考试里数“收丝挂钩四步走完的局”；旁观的老师按它把自己的阶段往前对齐。

## info 里常用的键

| 键 | 是什么 |
| --- | --- |
| `phase`、`max_phase` | 这一步的段号、本局到过的最高段 |
| `events` | 这一步的事件计数：进错血管、脱出口、第一档 |
| `is_success`、`hard_failure`、`failure_reason` | 第二档成功；硬失败和原因 |
| `tier1` | 本局拿过第一档没有 |
| `progress_ratio` | 两把尺子已走的路占全程多少，0 到 1 |
| `landmark_grid`、`landmark_grids` | 地标格和总格数 |
| `d` | 导丝减导管的送入长度，mm |
| `policy_action`、`executed_action` | 策略给的动作、仿真实际执行的（都按速度上限归一） |
| `reward_materials` | 这一步的奖励原料 |
| `expert_action` 等 | 老师的答案（打开旁观时才有） |
| `wall_breached` | 这一步中心线穿出了血管网格 |
| `snapshot_saved`、`snapshot_start` | 这一步存了一份存档；这一局是从哪份存档开的 |
| `labels`、`label_details`、`raw_action` | 框架原有：标签值、标签细节、执行的原始速度 |

完整的键看各个 `…Info` 类。

## 开局（`WireFirst`）

每局开始、策略走第一步之前：两根器械回到插入点，再只推导丝，全速（一步 5.28 mm：一个控制步是 33 个 4 ms 的小步），最后一步放慢，8 步停在 40 mm；导管停在插入点，钩还没出来。这 8 步在 stEVE 的 `start` 里走，在观测、奖励、裁判重置之前，不算这一局的步数；走完把“执行过的动作”清零，策略看到的第一帧和没动过一样。`configs/two_devices_real_mass.yaml` 起的配置都用它，第一轮配置用的是 stEVE 的 `InsertionPoint`（两根都在 0）。

为什么（2026-11-04 实测，`docs/simulator_changes.md` 问题 23）：插入点在降主动脉中线上（那里血管半径约 11.2 mm），器械杆身在插入点被固定在中间；钩（130°、半径 8 mm、加 8 mm 直头）向一侧伸出约 19 mm，钩在插入点 40 mm 以内出来就张不开、压在壁上（3–7 N，撞上那一下 9 N）。随机推拉 10 局 × 300 步，导管头在 40 mm 以内的 535 步里 124 步超过 2 N；先送导丝以后 325 步里 12 步。之后导丝领不领先由策略自己管，不加规则（用户定，2026-11-04）。

## 仿真存档（`snapshot.py`）

存的是什么：BeamAdapter 每个小步都按两根器械的送入长度重新排节点，所以要存的只有三样——推进控制器的送入长度、转角、`indexFirstNode`；梁每个节点的位置（含朝向）和速度。裁判的经历（段、事件累计、计数）一起存。

怎么恢复：新场景 `reset` 之后把控制器的三样设成存档值，再加一个一次性的 SOFA 控制器，在第一个小步控制器排好节点之后、碰撞检测之前把节点位置和速度整个覆盖回去，然后这个小步照常算一次物理（零动作）。恢复以后裁判保留存档里的经历，其余的量按这一局自己的目标重新量。观测里带记忆的项（最近几步动作、前几帧）从头开始。

两种用法：
- 训练器指定：`env.reset(options={"snapshot": 路径})`，起点库用这个。训练局沿途存档由 `EpisodeSnapshots` 做，每局第一次走进新的一格就存一份，`step` 回来的 `info` 带 `snapshot_saved`。
- 固定库按概率抽：`env.start` 写 `FromSnapshot`，每局按 `fraction` 从库里抽一份，其余局走普通开局。`count_saved_steps: true` 时，从存档开的局步数上限扣掉存档前已走的步数。

限制：存档记着物理步长和器械清单（名字、长度），对不上就拒绝恢复；节点数对不上也拒绝。存档只能在存它的那根血管上恢复。

验证在 `tests/test_snapshot.py`；多种子多接手点的验证见 [`docs/simulation_speed.md`](../../../docs/simulation_speed.md)“开局存档”一节。

## 标签

| 名字 | 是什么 |
| --- | --- |
| `distal_force` | 导丝头最后 3 mm 的接触力（一步内的最大值，N）。名字沿用 G13，受力预测头不用改 |
| `catheter_tip_force` | 导管头最后 3 mm 的接触力 |
| `wire_tip_clearance`、`catheter_tip_clearance` | 两头离壁距离（mm，在血管里为正），用中心线管子算 |

两根器械在 BeamAdapter 里是一根共用节点的梁，碰撞链的末端是领先的那个头。较短的那根头在“领先的头往回数 x mm”处（x = 两根送入长度差，每个物理小步重读），它的最后 3 mm 就是 [x, x + 3] mm，在这里的接触按参考代码的方法定位、相加。领先的那根算出来就是 G13 的 `DistalForce`（有测试逐位比对）。

## 手写推法（`script.py`）

用途：前缀、示范、旁观的老师、考试的参照、奖励校验。它读仿真真值，只在训练时用，部署时没有。

每个阶段做什么：

| 阶段 | 做什么 | 完成 / 补救 |
| --- | --- | --- |
| 过弓 | 导丝领先 40 mm，两根一起过弓 | 导管头过头臂干口 33 mm |
| 摆钩 | 导丝退到钩后 37 mm，钩朝下，钩底拉到口上方 4 mm | 到位；超时换下一个偏移 |
| 转钩 | 钩原地转朝上，瞄 −30°、容差 20°（软钩翻过顶后停在约 −50°，瞄 0° 永远等不到） | 稳 3 步；超时就重摆 |
| 拉进、坐稳 | 3 mm/s 回拉，钩尖进头臂干 2 mm 再拉到 20 mm（再拉只到 23，拉 35 mm 会整根出口） | 深度到；拉空、进错血管就重摆 |
| 出管 | 导丝 30 mm/s 出到领先 16 mm，导管不动（推导管只会把梁在弓里压弯） | 领先到 |
| 瞄准 | 从导丝折线读软头弯的方向，转导丝到背离右锁骨下 | 对准或超时 |
| 快推 | 25 mm/s 冲头臂干末端；进不动、往回退、导管头后退都退 1 步再冲 | 导丝进右颈总 6 mm；进锁骨下太深就退回头臂干再瞄再冲 |
| 上颈总 | 10 mm/s，导管只在滑出 8 mm 以内时慢拉 | 到 50 mm 交给颈内段 |
| 颈内段 | 向下的侧支前后瞄背离它，分叉下 20 mm 起瞄颈内一侧、对准才慢推；进颈外太深就退到分叉下再瞄 | 进颈内够深算锚住；不进就退一点；超时重坐 |
| 跟进 | 导管顺导丝走（末端、分叉前后放慢）；导丝底部不动；不进就“推 3 停 3” | 到导管目标前 2 mm；导丝滑回分叉附近就停推、回颈内段重锚 |
| 收尾 | 导丝修到目标，导管修到目标深度 | 裁判的条件保持住 |

每个数和补救的细节在 `script.py` 的 `ScriptParams` 和文件开头的说明里。2026-11-14 加了几个默认关的参数，给旁观的老师用（钩在原地转朝上、永远朝同一个方向转、旁观时不认输等），见 `docs/training_framework.md`。

这套仿真里量到的脾气（2026-11-06）：两根器械在 BeamAdapter 里是一条链，长度由领先那根定；导丝头顶住还推，链在弓里压弯、两头一起退；推落后的导管不加长链、沿导丝 1:1 走，所以导管只在导丝头自由或锚住时能前进。导丝 30 mm 软头在头臂干末端前 9–13 mm 贴壁停住，慢推、各种转都过不去，25 mm/s 短促快推能过（憋在弓里的长度一下放出来，导丝头一步跳 10 mm，和真实手术里“弹进分支”是一回事）；进哪支由软头弯的朝向定。

成功率：默认参数在种子 1000–1019 上 20 局成 17 局（2026-11-07、11-10 两次），失败的 3 局是同一种走法：导丝头到了头臂干末端进不了任何一支。每次验收和重录的数在 `agent_log/TRAINING_RECORD.md` 和工作日志里。

## 两个容易踩的地方

1. **推送时节点折线比送入长度短**，较短那根的头要从领先的头往回数。stEVE 的 `instrument_trackings3d` 从根部往前数，推送时较短那根的头会偏几毫米。短的长度 = 推送速度 × 阻尼系数（stEVE 写死 0.2 s）：满速 40 mm/s 短 8.0 mm，停下约 3 步追平。是阻尼拖住伸长的速度，不是器械真的软（2026-11-03 实测）。
2. **stEVE 的 `TrackingOnly.tracking3d_space` 在 C 臂角度大时建不出来**：它把血管坐标框的两个角旋转后直接当上下界，有的维 low > high。所以目标用自己写的 `TaskTarget2D`；框架那一边的修正见 `docs/simulator_changes.md` 问题 21。

## 测试

```bash
conda run --no-capture-output -n guidewire_rl python -m pytest -q tests/test_two_device_env.py -m "not sofa"   # 不起 SOFA，几秒
conda run --no-capture-output -n guidewire_rl python -m pytest -q tests/test_two_device_env.py                  # 含 SOFA
```

| 改了什么 | 跑哪个 |
| --- | --- |
| 尺子、裁判、观测、奖励、动作 | `test_two_device_env.py`、`test_two_device_round2.py` |
| 手写推法 | `test_two_device_script.py`（含一整局 SOFA）；第一轮版 `test_two_device_script_round1.py` |
| 旁观的老师 | `test_shadow_expert.py` |
| 存档、起点库 | `test_snapshot.py`、`test_start_library.py` |
| 离壁、穿墙 | `test_wall_monitor.py` |
| 受力标签、力爆掉的截断 | `test_tip_force_fast.py`、`test_force_blowup.py` |
| 停住就结束 | `test_stall_end.py`（含真仿真器一条）；训练器那边在 `test_parallel.py` |
| 训练器那边（课程、日程、回放） | `test_round2_training.py`、`test_round2_schedule.py`、`test_two_device_training.py` |

起 SOFA 的两器械测试共用 `tests/two_device_scenes.py`（建场景、推器械、读碰撞链）。
