# 训练框架怎么用

代码在 `guidewire_rl/`，一个实验 = `configs/` 里一份 YAML。

这份只讲怎么用：怎么跑、怎么改、每个配置键是什么意思。为什么这样设计、试过哪些不行，在 [`design.md`](design.md)；每次训练的结果在 `agent_log/TRAINING_RECORD.md`。

## 两个环境

| conda 环境 | 仿真器 | 配置 | 用来 |
| --- | --- | --- | --- |
| `guidewire_rl` | SOFA 25.12 + 工作区 `stEVE/`（上游）+ 我们的修正 | `configs/g13_upstream.yaml`（合成血管）、`configs/phantom_g13.yaml`（lab 的 phantom）、`configs/phantom_friction.yaml`（phantom，摩擦 0.1）、`configs/phantom_root.yaml`（再把主动脉中心线延到根部的底）、`configs/phantom_all_vessels.yaml`（再给 phantom 每条血管都配中心线）、`configs/phantom_real_mass.yaml`（phantom，导丝真实质量 + 2 ms 物理步 + 碰撞预判 2 mm）、`configs/two_devices_real_mass.yaml`（导丝 + 钩形导管，两根都按真实质量，每根器械按自己的半径碰墙）、`configs/two_devices_ica.yaml` 和继承它的五份（两器械到右颈内，现在在用，见下面“训练配置”一节） | **新的训练都用这个** |
| `contact_aware_tdmpc2` | SOFA 23.06 + Wilson 自带的 eve | `configs/g13.yaml` | 复现、对照 G13 |

两个环境装的是同一份 `guidewire_rl` 代码。`configs/g13_upstream.yaml` 继承 `g13.yaml`，只把导丝换成上游的 Angled。`configs/phantom_g13.yaml` 再继承它，只换两处：血管换成 lab 的 phantom（`BiroPhantomArch`，数据和已知问题见 `guidewire_rl/env/vessels/biro_phantom/README.md`），血管几何量改用 `frame: vessel`。`configs/phantom_friction.yaml` 继承 `phantom_g13.yaml`，只把导丝和管壁之间的摩擦系数从 0.001 改成 0.1（2026-10-27 用户决定，来历见该文件注释）。`configs/phantom_root.yaml` 继承 `phantom_friction.yaml`，只把血管换成主动脉中心线延到根部底的版本（`BiroPhantomArch(centerline: root_extended)`，碰撞网格不变；为什么和怎么做见 `biro_phantom/README.md`）。phantom 的血管还可以按整套版本选：`vessel_tree: {type: BiroPhantomArch, version: rebuilt_cut | spheres_cut | lab_cut | rebuilt | spheres | lab}`（`rebuilt*` 模型内壁版、`spheres*` 内接球版，都只用 STL 生成；`lab*` 是 lab 输入做的内接球版，留着对照；`_cut` 只有头和胸腔 / 不带后缀是完整带右臂；是什么、怎么来的见 `guidewire_rl/env/vessels/biro_phantom/README.md`，2026-10-28/29 做的，还没有配置用它们）。`configs/phantom_all_vessels.yaml` 继承 `phantom_root.yaml`，只把血管换成每条血管都有中心线的版本（`centerline: all_vessels`）：目标点和 2D 坐标范围仍只按原来 6 条，离壁距离、奖励里的路程、stEVE 停推的位置看全部 75 条。`configs/phantom_real_mass.yaml` 继承 `phantom_friction.yaml`，把导丝密度改成镍钛的真实值（stEVE 的数是单位写错，重约 3300 倍）、物理步 6 → 2 ms、碰撞预判范围 0.5 → 2 mm、受力标签每控制步 66 个小步（2026-10-28 用户决定；为什么必须一起改、代价多少见 `simulator_changes.md` 问题 20）。`configs/two_devices_real_mass.yaml` 继承 `two_devices_round1.yaml`，两根器械改成真实质量（导丝同上，导管 1.2e-9 是按塑料约 1.2 g/cm³ 估的）、物理步 4 ms、碰撞预判 1 mm（2026-11-09 提速时从 2 ms、2 mm 改的，见 [`simulation_speed.md`](simulation_speed.md)），导管保留钩形、刚度换成真实导管的（E 4000 → 660 MPa），仿真器打开四个开关：每根器械一份碰撞点、按自己的外半径碰墙（`instrument_collision`，问题 8）、最短梁 0.05 mm（`min_beam_length`，问题 22）、两根重叠处按更密的那根排节点（`densest_sampling`，问题 23）、某个小步出现 NaN 就停下这一步（`stop_on_nan`）；最短梁和按密的排节点要我们编的 BeamAdapter。两个受力标签每控制步 33 个小步（4 ms 时）、读各自器械的碰撞点（2026-11-03 用户决定）。两套仿真器在物理上的差别，以及我们改了仿真器的哪些地方，见 [`simulator_changes.md`](simulator_changes.md)。

## 结构

| 位置 | 管什么 | 来源 |
| --- | --- | --- |
| `guidewire_rl/env/` | 按 YAML 的 `env:` 一节拼环境：仿真场景、观测、奖励、终止/截断、label | stEVE 自带零件 + 我们两个：`VesselGeometry`（G13 的 8 个几何量）、`DistalForce`（尖端接触力） |
| `simulator_modifications/`（工作区根目录） | 我们对仿真器的所有修改：导管空心、带种子重置（用 stEVE 原类的名字登记，只在上游 stEVE 下生效），以及 BeamAdapter 的补丁和编译脚本 | 见那里的 README 和 `simulator_changes.md` |
| `guidewire_rl/model/` | 世界模型、规划、训练步骤、额外预测头 | 世界模型和规划从参考包原样复制；额外预测头是新加的 |
| `guidewire_rl/train/` | 回放、训练主循环（一个环境 `trainer.py`；多个环境并行 `parallel_trainer.py` + `env_pool.py`）、存档、评估 | 照官方 TD-MPC2 的写法；并行是我们加的 |
| `scripts/` | `train.py` 训练、`evaluate.py` 评估、`inspect_env.py` 看观测 | — |
| `scripts/remote/` | 把一台新机器装成训练机：在笔记本上 `bash scripts/remote/provision.sh <主机> [端口]`（用法见 [`scripts/remote/README.md`](../scripts/remote/README.md)） | — |
| `runs/<名字>/` | 训练产物：`config.yaml`、`*.jsonl` 日志（四条流各记什么、wandb 该报什么、曲线怎么读见 [`wandb_metrics.md`](wandb_metrics.md)）、`checkpoints/` | — |

### `train/` 和 `model/` 里每个文件管什么

| 文件 | 管什么 | 对应的配置 |
| --- | --- | --- |
| `train/trainer.py` | 训练主循环（一个环境） | `training` |
| `train/parallel_trainer.py` | 多个环境一起走的主循环，等齐和不等齐两种 | `training.num_envs`、`async_envs` |
| `train/env_pool.py` | 每个环境一个进程的进程池；崩了、卡了、占内存太多就换新的 | `training.step_timeout_s`、`worker_max_rss_mb` |
| `train/setup.py` | 把配置和环境接成一个能开跑的训练器（新开、续训、从 G13 存档起）；回放抽样时重算奖励也在这里接 | — |
| `train/replay.py` | 回放：每步存观测、动作、奖励原料、来源、老师的答案；示范的保底份额 | `replay`、`training.demos` |
| `train/demos.py` | 把示范文件读进回放 | `training.demos` |
| `train/curriculum.py` | 起点课程：这一局从哪开、前沿怎么挪 | `training.start_curriculum` |
| `train/start_library.py` | 起点库：存档按格收、抽、淘汰 | `training.start_curriculum` 的 `supply`、`library` |
| `train/schedule.py` | 日程：按考试结果撤示范、升受力系数、停训 | `training.schedule` |
| `train/evaluate.py` | 考试：固定种子、不加探索 | `evaluation` |
| `train/metrics.py` | 从环境的 info 里读每局的指标，做滚动汇总 | — |
| `train/logger.py`、`train/wandb_sink.py` | 把记录写成 jsonl；转发到 wandb | `logging` |
| `train/checkpoint.py` | 存档：模型、优化器、回放、计数、随机数状态 | `checkpoint` |
| `model/agent.py` | TD-MPC2 的更新步和规划；模仿、抗抖动、策略输出的限制这几项损失也在这里 | `training`、`planning`、`model` |
| `model/world_model.py` | 世界模型的网络 | `model` |
| `model/aux_heads.py` | 额外的预测头 | `aux_heads` |

环境那一边的零件表在各自文件夹的 README 里（两器械的在 `guidewire_rl/env/two_device/README.md`）。

## 改东西

改 YAML，不改代码。观测维度、模型输入层会自动跟着变；改了观测就不能接着旧 checkpoint 训。

- **观测**：`env.observation` 每行一项，按顺序拼起来。删一行、加一行，或改 `VesselGeometry` 的 `features`（`clearance` `heading` `bend` `radius` `valid` 任选）。`frame: reference` 是 G13 原样，`frame: vessel` 是修正坐标系后的离壁距离。
- **奖励 / 终止 / 截断**：`env.reward` 等，每行一个 stEVE 零件，多行自动相加或合并。
- **一局怎么结束**（两器械）：成功和硬失败是真结束（裁判判，`terminated`，价值不往后估）；超时 `MaxSteps`、数值崩 `SimError`、穿墙 `WallBreach` 是截断。穿墙 = 某根器械的中心线出了血管网格，按仿真器的故障处理（2026-11-09 用户定）：`WallInfo` 把它写进 `info["wall_breached"]`，训练器把那一步丢掉、这局按 `breach` 收尾（`episodes.jsonl` 的 `end`，`metrics.jsonl` 的 `breach_rate` 和 `wall_breaches`，评估报告的 `end_reasons`），不给 −1，和数值崩分开计数；"别顶墙"由受力扣分教。裁判的"头出中心线管子 6 mm"硬失败留着当保底，正常情况下穿墙先触发。
- **label**：`env.labels`，仿真每步量的、控制器看不到的量，只当训练答案。
- **额外预测头**：`aux_heads`，每个头指定答案用哪个 label、损失写法、权重，以及 `gradient`（`shared` 会影响模型，`detached` 只在旁边学）。
- **新实验**：新建一份 YAML，第一行写 `extends: g13.yaml`，只写要改的部分。字典逐项合并，列表整个替换；写 `null` 就是删掉继承来的那一项，比如换一种血管时，用 `vessel_tree: {type: BiroPhantomArch, seed: null, scaling_xyzd: null}` 去掉 `AorticArch` 的参数。
- **临时改一个数**：命令行加 `--set 段.键=值`，例如 `--set training.total_steps=1000`。只能改配置里已有的键。路径走到一个列表时，下一段写那一项的 `name`，例如 `--set env.monitors.phase_detector.thresholds.target_mm=3.0` 改的是 `env.monitors` 里名字叫 `phase_detector` 的那一项。
- **折扣因子 γ**：模型按 TD-MPC2 原版的经验公式算 γ = 1 − `discount_denom` ÷ 步数上限（`MaxSteps`），再夹在 `discount_min` 和 `discount_max` 之间（`training` 段，默认 5 / 0.95 / 0.995）。所以改 `MaxSteps` 会顺带改 γ。要钉住就把 `discount_min` 和 `discount_max` 写成同一个数；两器械颈内配置（`two_devices_ica.yaml`）钉在 0.99375（2026-11-09 用户定）。
- **C 臂角度**：每份配置默认在 `env.scene.fluoroscopy.image_rot_zx` 写死一个角度，整次训练都用它。要让每一局按目标点所在的分支用那个分支自己的角度，配置里加一行 `env: {scene: {fluoroscopy: {image_rot_zx: per_target}}}`（或命令行 `--set env.scene.fluoroscopy.image_rot_zx=per_target`）。之后每局抽完目标点、算第一帧观测之前，从这个 phantom 版本的角度表（`guidewire_rl/env/c_arm/data/`，选角度脚本事先写好，训练时不重算）里查出这个分支的角度换上。每个分支的角度是按从主动脉过来的整条路线选的（2026-11-03 用户定）：分叉口前 60 mm 主动脉 + 路上经过的每条血管 + 目标分支，比如右颈总是主动脉 + 头臂干 + 右颈总，两器械的目标用它，模型内壁版上是 LAO 15°、头位 7.5°（stEVE [52.0, −12.2]）；目标点在某个目标分支往后接着的那一段上（例如右颈总后面的 `rcca_rest`）时用那个目标分支的角度，在五个目标分支（头臂干、右颈总、右锁骨下、左颈总、左锁骨下）以外就报错。现在有两份角度表：模型内壁版去掉右臂的（`version: rebuilt_cut`，两器械配置用的这个）和 lab 内接球版全部血管的（`version: lab` 或 `centerline: all_vessels`）；别的版本建环境时报错，并给出生成角度表的命令。单导丝的 phantom 配置也能打开：stEVE 换算 X 光坐标范围只转两个角、这些角度下建环境会报错，已经修了（`simulator_changes.md` 问题 21，`simulator_modifications/fluoroscopy.py`）。细节见 `guidewire_rl/env/c_arm/README.md` 的 "In training"。

改完先看一眼控制器会看到什么：

```bash
conda run --no-capture-output -n guidewire_rl python scripts/inspect_env.py --config configs/g13_upstream.yaml
```

场景里的导丝写在 `instruments:` 下（上游的叫法）；旧版 eve 的配置写 `devices:`，两种都认。

要加一种新的观测或 label，写一个类，用 `@register("observation")` / `@register("label")` 登记，YAML 里就能按名字用。

## 跑

下面以新环境为例；复现 G13 时把环境换成 `contact_aware_tdmpc2`、配置换成 `configs/g13.yaml`。

```bash
# 测试：快速的
conda run --no-capture-output -n guidewire_rl python -m pytest -q -m "not sofa and not checkpoint"
# 测试：全部（含真实 SOFA 场景；主力机空着约 19 分钟，和训练同时跑会互相拖慢）。
#   guidewire_rl 跑仿真器修正（upstream），跳过和参考包对照的（reference）。
conda run --no-capture-output -n guidewire_rl python -m pytest -q
# 旧环境那一套（和参考包逐位对照）2026-11-15 起退役，平时不跑；要回头对照 G13 时在主力机上跑：
#   conda run --no-capture-output -n contact_aware_tdmpc2 python -m pytest -q
# 训练
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/g13_upstream.yaml
# 断了接着训
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/g13_upstream.yaml --resume runs/<名字>/checkpoints/step_N.pt
# 从 G13 的 checkpoint 接着训（连它的回放一起）。G13 的回放来自旧仿真器，所以这条只在旧环境用过：
conda run --no-capture-output -n contact_aware_tdmpc2 python scripts/train.py --config configs/g13.yaml --init-from-reference reference/checkpoints_g13/step_260000.pt
# 评估
conda run --no-capture-output -n guidewire_rl python scripts/evaluate.py --config configs/g13_upstream.yaml --checkpoint <checkpoint>
```

`scripts/evaluate.py` 默认出均值动作（不加探索）。加 `--sample` 就像训练时那样出动作（策略网络抽样 / 规划器带噪声）：同一个存档、同一批种子各考一遍，可以看出策略是不是不加噪声就会停住。`--save-episodes <目录>` 存逐步记录，`--workers N` 开 N 个进程。

### 跑正式训练时要知道的

- 仿真开 20 个最合适（`--set training.num_envs=20 --set training.async_envs=true`）；开到 26 个不会更快。
- 后台起训练用 `env --default-signal=INT nohup conda run …`，否则发中断信号停不下来。当前这次训练的完整命令在 `agent_log/CURRENT_STATE.md` 第 6 节。
- 查训练还在不在跑：`pgrep -af scripts/train.py`；看它跑到哪了：`tail -1 runs/<名字>/metrics.jsonl`。
- 训练在跑的时候，不要原地改它会导入的代码和它用的配置（规矩和原因在共享规则“开发与验证”）。要改就先停，或者另开一个工作树。
- 主配置里考试写的是每次 50 局；正式训练在命令行或子配置里改成 20 局。20 局只能看大概，定论要 100 局（原因见 `design.md` 第 11 节）。
- wandb 默认在线、只送核心视图的曲线，其余的看本地 jsonl。

## 多个环境一起跑

`training.num_envs` 默认 1。设成 N（命令行加 `--set training.num_envs=12`），就开 N 个进程，每个进程一个 SOFA 场景，同时走：

```bash
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/g13_upstream.yaml --set training.num_envs=12
```

训练节奏和一个环境时一样，步数按所有环境加起来算：前 `seed_steps` 步随机动作，攒够后一次更新 `initial_updates` 次，之后每走一步更新 `updates_per_step` 次；存档、评估、打印都落在原来的步数上（`step_20000.pt` 还是 `step_20000.pt`）。不一样的地方：

- 每一轮，所有环境的动作由一次规划一起算出，每个环境用自己上一步的规划结果当起点。
- 这一轮该做的更新，在仿真走的同时在 GPU 上做，用的是到上一轮为止的数据，所以策略最多落后一轮（N 步）。存档和评估前先补齐，所以第 S 步的存档里更新次数和一个环境时一样。
- 评估在另开的一组环境里跑（个数 = 评估局数和 N 里小的那个），到步数马上跑，不打断训练里正在走的局；这组进程每次评估时开、评估完就关（2026-11-02 前一直开着，两器械时 10 个约 7 GB）。每一局用自己的随机数流和自己的规划起点，所以评估结果和一个环境时逐位相同，和这组环境的个数无关（有测试，假环境和真实 SOFA 场景都比过）。
- 某个环境报错：只丢那一步，那局记为出错，下一轮重新开局，和一个环境时一样。进程崩了（SOFA 段错误）或 `training.step_timeout_s` 秒（默认 120）没回音：换一个新进程接着跑，训练不停。都记在 `events.jsonl`（`dropped_step` 带 `env` 编号，换进程记 `worker_restarted`）。
- **仿真进程定期换新**（`training.worker_max_rss_mb`，单位 MB，默认不设 = 从不换；`two_devices_round1.yaml` 设 1500）：SOFA 每局拆场景时留下一部分内存不还（[`simulator_changes.md`](simulator_changes.md)"场景越跑越慢"：两器械每走 1000 步多占约 24 MB，一直直线涨），只有进程退出才收得回来。设了这项，某个仿真进程一局走完、要开下一局时，主进程先读它占的内存，超过就让它退出，在原位置开一个新进程（加载 SOFA、按同一份配置建场景，两器械实测 2.4 s，这段时间主进程停着），再把这局的种子发给新进程。只在两局之间换，不切断任何一局；那局的数据早已在主进程里。新进程跑出来的局和老进程在浮点末位上不同（同一种子、同一推法 150 步，导丝头位置最多差 0.06 mm；新进程之间逐位相同，2026-11-02 实测），20 个进程本来就彼此不同，所以对训练没有新影响。按 1500 MB：新进程约 760 MB，每个进程约 3 万步换一次。记在 `events.jsonl`（`worker_recycled`，带换前内存和用时），计数 `env_recycles`，和崩溃换进程的 `env_restarts` 分开。
- 每个仿真进程启动时把自己的 `oom_score_adj` 调到 500（调高不需要权限）：机器内存用完时，内核先杀占内存最多的仿真进程（丢一局，进程池换新的），而不是训练主进程（整个训练没了）或桌面端（它自己的进程设 200–300）。2026-11-01 run A3 就是内核连杀 7 次桌面端仍不够，最后杀了主进程。
- `metrics.jsonl` 每行带 `memory_mb`：主进程、训练仿真进程里最大的和合计、机器还剩多少可用（MB）；控制台每行下面多一行同样的数和已换进程数。
- 主进程（规划和更新）自动独占最快的一个核（这台 i9-14900K 上是 CPU 8、9），仿真进程用其余 30 个。主进程卡在单线程速度上：和仿真挤一个核或被排到小核上时，一次更新从 8.5 ms 慢到 13–23 ms。启动时控制台会打印分到哪几个 CPU。
- 控制台每行下面多一行：主进程的时间里，规划、更新、等仿真各占多少。等仿真占大头，加环境还能更快；更新占大头，再加就没用了。
- 续训时 N 可以和原来不一样。
- **不等齐**（`training.async_envs: true`，默认 false）：上面的做法每一轮要等 N 个环境都走完，最慢的那个（小核上、正碰壁的）决定一轮多长。打开这项后，哪个环境走完就马上给它算下一步、发出去，更新一次一次地夹在中间做。步数在结果回来时计数，所以存档、评估、打印仍落在整步数上。和等齐的差别：同一配置跑两次结果不再逐位相同（谁先回来看时机）；每步用的模型落后多少不固定（几次到这一步耗时内做的更新数）；到 `seed_steps` 时还在走的那几步是按随机动作发出去的（随机步最多多 N 步）；更新欠到两轮（2N 次）以上时暂停发新步，等补上，保证每步一次更新的比例；到 `total_steps` 时正在走的步丢掉。实测（2026-10-28，绑核，每秒步数）：真实质量 24 个环境 14.7 → 25.9（30 万步约 3.2 h），16 个 12.2 → 22.3；现在的质量 24 个 73.0 → 80.6。真实质量下再加环境没用：24 个 SOFA 进程已把 30 个逻辑核占满，每步平均拉长到约 0.9 s；现在的质量下不等齐也没用，主进程被更新占满（86 %）。
- 别人怎么做（2026-10-28 查的，细节见工作日志当天条目）：TD-MPC2 官方 `main` 只有单环境；实验分支 `vectorized_env` 用 gymnasium 的 `AsyncVectorEnv`（每个环境一个进程，一起走一步，一次批量规划，再串行做 k 次更新），但要求所有环境同时结束、规划起点不分环境，而且作者说它和终止信号（`episodic`，我们的配置在用）不兼容、不打算合并。ManiSkill 的 TD-MPC2 基线同一写法，靠所有局等长绕开。tdmpc2-jax 像我们一样按环境保存规划起点、局结束时单独清零。stEVE 作者自己的 `stEVE_rl`（lab 复现论文基线用的）是另一条路：每个进程一份 CPU 策略副本、各自跑整局、每 100 局同步一次权重、每 20 步更新一次；它的"进程崩了就换一个、卡住就超时重启"我们照做了。我们这份和这些的差别只有一处：更新和仿真走步重叠，所以策略最多落后 N 步；上面这些都没做重叠（DreamerV3 的全异步版和 Sample Factory 则接受更大的落后）。gymnasium / SB3 的向量环境都没有"只走一部分环境""进程崩了换新的""绑核"，所以进程池是自己写的；同时具备这三项的现成库只有 Ray RLlib。
- 第一次集中更新要等回放里有走完的一局（一个环境时也是这样）。N 个环境时，前 `seed_steps` 步里每个环境只走 `seed_steps / N` 步，这个数小于一局最长的 200 步时，可能一局都没走完，集中更新就往后推，推多少步就少多少次更新。G13 的 5000 步在 N ≤ 25 时够用。

## 测试锁住了什么

- `configs/g13.yaml` 的数和 G13 checkpoint 里存的训练配置一致。
- 用 `g13.yaml` 拼出来的环境，同一个种子下观测、奖励、受力标签和参考包逐位相同（只在旧环境）。
- 复制过来的模型读 G13 checkpoint：规划出的动作、受力预测，以及受力权重为 0 时的一次训练更新，都和参考包逐位相同（只在旧环境）。
- 仿真器修改（只在新环境，`tests/test_simulator_modifications.py`）：推进正好 0.01 mm 不崩溃且加载的是我们编译的 BeamAdapter；导管按空心管弯；三种目标点或血管收到 numpy 种子能重置且可复现；打开受力标签不改变动力学；X 光坐标范围（问题 21）在角度表的每个角度下都是包住 8 个角的盒子、不改观测数值，`TrackingOnly` 只有一个模块登记。
- phantom（`tests/test_biro_phantom.py`）：复制来的两份数据和 lab 的文件逐字节相同；建出来的血管逐字段等于 lab 交给 `FromMesh` 的内容；`phantom_g13.yaml` 相对 `g13_upstream.yaml` 只改了血管和 `frame`；phantom 环境里实际走几十步，22 个值在截断前都在 [-1, 1] 内（只在新环境）；延长版中心线：文件哈希、只改了主动脉末端、新点的半径等于到碰撞网格的距离且在横截面上居中、分叉点和目标候选点等不变、stEVE 停推的位置移到根部底；全部血管版：文件哈希、6 条不变、加的分支来自 lab 血管、管子盖住碰撞网格内腔、目标点和 2D 范围不变、目标分支截断处不再停推。
- C 臂按目标选角度（`tests/test_c_arm.py`、`tests/test_c_arm_per_target.py`）：路线是从主动脉过来的整条（右颈总 = 主动脉 60 mm + 头臂干 + 右颈总），头臂干、左颈总、左锁骨下的路线和改之前逐位相同，改之前的路线定义（`--route own_ostium`）还能用、和原来的数逐位相同，分段的分数合起来就是整条路线的分数；两份角度表是从现在的中心线文件生成的（哈希一致），里面的角度就是 README 记的那些，每个目标都有分段（主动脉、头臂干、目标分支）的分数；加 `--run-slow` 时把模型内壁版的整套搜索重跑一遍（约 2 分钟），结果和存的表逐项相同；没有表、表和中心线文件对不上、目标不在五个分支上，都报错并说要跑什么命令；lab phantom 上用 G13 的五个目标分支，每局换分支时角度跟着换，目标点和把 C 臂直接固定在那个角度时逐位相同；两器械的目标落在右颈总前后两段上都用右颈总的角度；两器械环境打开开关和直接写死右颈总的角度，观测（目标点和路口图逐位，其余到 1e-6）、奖励、2D 范围相同（真实 SOFA，只在新环境）；单导丝的 `phantom_all_vessels.yaml` 打开开关能建能跑，每局角度跟着目标分支（真实 SOFA）；不开开关的配置建出来和原来一样。
- 回放取的片段连续且不跨段；训练主循环的步数安排、存档、续训、仿真报错后继续。
- 并行（`tests/test_parallel.py`，用假环境起真进程）：环境报错后进程还在；进程崩了、卡住了会换新的；一个环境时一起规划和原来的规划结果相同；几个环境一起规划时，每个环境的结果和单独规划它时相同；并行训练的步数、更新次数（第 30 步存档里的更新次数和一个环境时一样）、存档、评估、续训；报错和进程崩溃时训练不停；同一配置跑两次，轨迹和评估逐位相同；进程超过内存上限时只在开新局前换、换的新进程同一种子给出同一局、走步中不换，每次开局都换新进程时训练轨迹、评估和更新次数与不换逐位相同；评估进程每次评估完都关掉；`metrics.jsonl` 带内存；仿真进程的 `oom_score_adj` 是 500；并行评估和单环境评估逐位相同（假环境 1/2/3 个进程；真实 SOFA 场景 2 个进程，2 局各 25 步，只在新环境：旧版 eve 不重建场景，同一个种子作为一个进程的第一局和第二局，开局导丝转角就不同，2026-10-28 量过）。

## 两器械训练用到的配置键

两器械训练的主配置是 `configs/two_devices_ica.yaml`，设计和理由在 [`design.md`](design.md)。它用到的键：

- `env.reward_formula`：奖励不再是 stEVE 奖励项相加，而是一份公式（`guidewire_rl/env/two_device/reward_formula.py`），环境在标签算完以后算出本步的分和原料向量（`info["reward_materials"]`）；`env.reward` 里放 `ZeroReward`。回放存原料，抽样时按配置里的系数重算（`train/setup.py` 的 `reward_recomputation`），所以改系数、开备用项不用清回放，但示范文件必须带 `materials`（`scripts/record_demos.py` 用第二轮配置重录）。
- `training.start_curriculum`：起点课程（`train/curriculum.py` 的 `StartCurriculum`）。每局抽"从插入点起"或某一格的进度比例，手写推法在环境进程里现跑前缀到那个比例再交给策略（`env/two_device/prefix.py`；`EnvPool` 的 reset 命令带 `prefix_ratio`）。前缀的步算在步数里、进回放（来源 `prefix`），`episodes.jsonl` 多了 `start_grid`、`start_ratio`、`prefix_steps`，`metrics.jsonl` 多了 `start_frontier`、`start_weights`、`start_rates`、`success_by_start_grid`。和 `training.cutoff_mixture` 互斥。
- `training.self_demos: {enabled, capacity}`：策略自己拿到第二档的局在回放里受保护（不淘汰），最多留 `capacity` 局；示范配额从脚本示范和这些局里一起抽。
- `aux_heads` 多了 `target_scale`：头预测 `标签 / target_scale`（离壁用 10，单位成厘米），`aux_predict` 乘回来。
- 检查点、评估、控制台行按"跨过了步数的整倍数"触发（`trainer.crossed`），因为一次 reset 带来的前缀是几百步一下子到账。
- 评估记录多了 `tier1`（第一档拿到没）、`progress_end`（结束时的进度比例）、`lead_end`（结束时的领先量）。

课程、日程和别的几个键：

- `training.start_curriculum.supply: self`（2026-11-12）：起点不靠脚本，用策略自己走到过的状态（`train/start_library.py` 的 `StartLibrary`）。每局训练局第一次走进一个新的 10 % 格时，环境进程存一份仿真存档（`env/two_device/snapshot.py` 的 `EpisodeSnapshots`，存到 `runs/<run>/starts/grid_KK/`，`step` 回来的 info 带 `snapshot_saved`），训练器按格收进库，每格只留最新 `library_cap`（100）份、多的连文件删掉；续训时从文件夹读回。课程只在存够 `library_min`（5）份的格里抽：练的格 = 前沿及以下有存档的最高格（`training_grid`；一个都没有时全从插入点开），前沿之上有存档的格给底数，前沿按原规则从练的格往回退；三成从插入点开。抽到的格从库里随机挑一份，reset 命令带 `snapshot`（`EnvPool`），环境 `reset(options={"snapshot": path})` 交给 `env.start: FromSnapshot`（这时不需要固定库，`fraction: 0` 时就是平常的开局）。恢复以后裁判保留存档里的经历（段、事件、计数，`phases.HISTORY_FIELDS`），其余按这一局自己的目标重新量。从存档开的局重新算 800 步（不从插入点算：走得远的状态多在第 600 步以后才到），`info["snapshot_start"]` 记从哪份、什么进度开的，按真实开局进度记格。`metrics.jsonl` 多 `start_library`（每格份数）、`start_training_grid`。配置 `configs/two_devices_ica_selfstart.yaml`。导丝出管以后、过了头臂干的状态恢复可靠（2026-11-12 量：21 个状态，恢复后同样动作 30 步导丝头差最大 0.8 mm、多数 < 0.3 mm，阶段全一致，0 仿真错误；WORK_LOG 当天条目）。
- `training.start_curriculum.grid`：`distance`（默认，进度比例每 10 % 一格）或 `landmarks`（按地标）。地标由裁判按路线结构自动生成，不写血管名（`PhaseDetector.make_landmarks`）：导管沿主动脉到第一个分支口、坐进它 8 mm；导丝过后面每个有血管分出去的地方（进路线拐进的血管 8 mm、过侧支口、过目标前最后一个分叉 8 mm）、到目标球；导管进后面每根路线血管 8 mm、到第一档深度、到 F。格 = 1 + 从第一个起连续成立的里程碑数，每步在 `info["landmark_grid"]`（总格数 `landmark_grids`）。这条路线九个里程碑 = 格 1–10，`bins` 必须等于 `landmark_grids`（第一次 reset 时查）。抽到格 k 时前缀走到 `landmark_grid ≥ k` 那一步交接（`run_prefix(key="landmark_grid")`，`EnvPool` 的 reset 命令带 `prefix_key`）。
- `training.schedule`：日程状态机（`train/schedule.py`），每次评估后走一步。三段：课程期（前沿回到插入点且最近两次评估合并成功率 ≥ `pass_rate`）→ 示范配额按 `demo_ramp_steps` 线性退到 0 → 收尾（第二档成功率连续 `stop_patience` 次评估没破纪录就停训）；可选第四段把提示分（进度分、跟进分）系数退到 0（`hint_ramp`，默认关）。每条坡都有守门：最近两次合并成功率比开坡时掉超过 `guard_drop` 或低于 `guard_floor`，坡停、退回上一次评估时的值、保持 `hold_evals` 次评估，回来了半速继续，再掉一次就永远停在当前值。梯子（`ladder`）：前沿格连续 `stuck_evals` 次评估成功率低于 `stuck_rate`、后一格高于 `next_rate`、前沿停了 `stuck_steps` 步以上，就开第一级专用提示分（奖励的摆钩点尺子 `arch_ruler`），该格过 `next_rate` 后关；第二、三级（顾问动作框、模仿顾问）要能从任意状态给出手写推法动作的"顾问"，没做，到时只记 `ladder_needs_advisor` 事件。示范配额和奖励系数改在回放上（`EpisodeReplay.set_demo_fraction`、换 `reward_fn` 后重算），环境里实时算的分仍按配置。事件在 `events.jsonl` 的 `schedule`，`metrics.jsonl` 多了 `schedule_stage`、`demo_fraction`、`hint_scale`、`ladder_level`。不写 `training.schedule` 时训练和以前一样。
  - 受力扣分先轻后重（为什么见 `design.md` 第 4 节）：`force_start`（两份带日程的配置都是 0.25）是受力扣分上限 `force_cap` 的系数，从第一批数据起生效；最近两次评估合并成功率到 `force_pass_rate`（不写就用 `pass_rate`，0.5）后，在 `force_ramp_steps`（15 万）步里线性升到 1，守门和别的坡一样（触发时系数退回上一次评估的值、保持、半速继续；第二次触发就停在当时的值，低于 1）。它和三段并行走，不算其中一段，可能和撤示范同时进行；收尾的停训规则在它还在升的时候不停。只改回放里的奖励（智能体学的那份）；环境实时算的分、每局记录、评估都按配置的全额。`metrics.jsonl` 多 `force_scale`，事件 `force_ramp_*`。`force_start: 1`（默认）= 没有这个系数。没有日程的配置（不带拐棍、自己的局当起点）目前没有这条坡。
  - 梯子默认是开的（`ladder: true`），但带日程的配置都写 `ladder: false`：过弓尺子不用。原因在 `design.md` 第 4 节“试过不行的”；要用它得先把“这个头到没到过摆钩点”（`hook_done_*`）两位加进观测。
- `evaluation.keep_best: true`：每次评估成功率创新高就存一份完整检查点 `checkpoints/best.pt`（覆盖上一份）。
- 备用项摆钩点尺子修过：每个头本局到过摆钩点 H 或进了头臂干以后改用普通尺子（`PhaseState.hook_done_*`）。修之前录的示范里摆钩点原料是旧算法，真要打开 `arch_ruler`（含梯子第一级）前要重录示范。
- 每局换场景参数：`env.randomize`（主配置写 `false` = 关，这样 `--set` 才能打开；`guidewire_rl/env/randomization.py`）。现在只有摩擦：`env.randomize.friction: {range: [低, 高], log: true, nominal_fraction: 0.3}`，每局开局前抽一个摩擦系数（`log: true` 是在对数上均匀抽），`nominal_fraction` 这么多的局留在 `env.scene.simulation.friction` 的值。另一组是执行和指令不一致：`env.randomize.execution: {delay_steps: [0, 4], gain: [0.7, 1.3], jitter: [0, 0.1], deadband_push_mm_s: [0, 1.0], deadband_rot_rad_s: [0, 0.1], nominal_fraction: 0.3}`，每局抽一次延迟（单位是一个控制步，可以带小数、可以跨多步；0.5 秒是 3.75 步）、每个通道的幅度倍数、抖动的幅度、推拉和旋转的死区，仿真执行的是改过的速度；每一项不写就是不改；范围两头写一样就是固定值（考试用）。策略的观测和记录里留的仍是发出去的指令。`env.info` 里加 `ScenarioInfo` 以后，每局的记录（训练的 `episodes.jsonl`、考试的逐局结果）多一项 `scenario: {friction: …}`。在一个固定的摩擦下考试：用老师旁观那一份配置加 `--set env.scene.simulation.friction=<数>`（不开 `randomize`）。
- 训练目标点连续取：`env.scene.target.sampling`，`grid`（默认，固定的候选点）或 `uniform`（训练用的点在 `depth_min_mm` 到 `depth_max_mm` 之间任取；和 `jitter_mm` 不能同时用；考试用的点不变）。
- 每局随机化的配置 `configs/two_devices_ica_randomized.yaml`（原名 `two_devices_ica_friction.yaml`）：继承老师旁观那一份，打开上面两项（摩擦和执行偏差两组都开），定期考试开“停住就结束”，并把 `round2_dagger_v5` 当时写在命令行上的设定（随机量上限、比考试严的训练线、不因成功率不涨而停）写进了配置。它的线比考试严，所以达标考试不用这一份配置，仍用老师旁观那一份。
- 路口地图（`JunctionMap2D`，`mode: all`）：过了目标的路口位置照列、方向两个数为 0；颈内任务 6 个路口都在目标之前，观测不变。

## 训练配置（开训时选一份）

下面五份的任务、观测、动作、奖励公式都一样，区别只在怎么帮策略越过靠乱试过不去的地方。另有一份 `configs/two_devices_ica_nodemo_best.yaml`（只奖励新进展）改了进度分的算法，已经弃用，留着对照。继承是一串：主配置 → 不带拐棍 → 自己的局当起点 → 示范当起点；只奖励新进展接在不带拐棍后面；老师旁观直接接在主配置后面。选哪种 = `--config` 给哪个文件。每种的想法、试下来怎么样，在 `design.md` 第 10 节和训练记录里。

| 方法 | 配置 | 起点从哪来 | 回放里的示范 | 训练里跑不跑手写推法 |
| --- | --- | --- | --- | --- |
| 不带拐棍 | `configs/two_devices_ica_nodemo.yaml` | 每局从插入点起 | 无 | 不跑 |
| 两根拐棍 | `configs/two_devices_ica.yaml` | `supply: prefix`：手写推法每局现场走一段前缀再交接 | `runs/demos_round2c`（和示范当起点同一批），25 %，按日程撤 | 每局都跑 |
| 自己的局当起点 | `configs/two_devices_ica_selfstart.yaml` | `supply: self`：策略自己走得远的局沿途存的状态（`runs/<run>/starts/`） | 无 | 不跑 |
| 示范当起点 | `configs/two_devices_ica_demostart.yaml` | `supply: demos`：手写推法成功示范沿途存的状态（固定库，`library:` 指的目录）；从存档开的局步数上限扣掉存档前已走的步数 | `runs/demos_round2c`，25 %，按日程撤（同两根拐棍的日程） | 不跑（只在开训前造一次库） |
| 老师旁观 | `configs/two_devices_ica_dagger.yaml` | 每局从插入点起，没有起点课程 | 示范目录写在配置里，25 %，按日程撤；先在示范上预训练 | 每局都在旁边跑，只出答案、不执行 |

```bash
# 都这样开，只换 --config 和 --run-name（20 个仿真一起跑）
conda run --no-capture-output -n guidewire_rl python scripts/train.py \
    --config configs/two_devices_ica_demostart.yaml --run-name <名字> \
    --set training.num_envs=20 --set training.async_envs=true
```

- 抗抖动：奖励多一项 `reward_formula.action_change`（主配置 0.005，另外几份都继承），每步扣 0.005 × 四个动作“这一步执行的动作 − 上一步执行的动作”的平方取平均；动作按速度上限归一到 [−1, 1]，从观测里 `last_actions` 最新那一组读，所以策略、世界模型的奖励预测和规划器看到的就是被扣的那个量。满速来回翻转每步扣 0.02，匀速不扣。原料多一列 `action_change`（15 列）；以前存的局（示范、续训的回放）由回放按各局的观测补算。
- 模仿手写推法：`training.bc_coef`（主配置 1.0）。策略网络的损失多一项：`bc_coef` × 策略网络的均值动作和回放里的动作之差的平方，只在手写推法走过的步上算（前缀步、示范步，`agent._bc_mask`），策略自己的局不模仿；随日程撤示范一起撤（`bc_scale` = 当前示范配额 / 起始配额）。**这个键写在主配置里，另外几份配置都继承**：不带示范的两种没有可模仿的步、实际不起作用；“示范当起点”有示范、所以也带模仿（要不要带还没定，不要就在它的配置里写 `bc_coef: 0`）；老师旁观的配置把模仿的对象换成老师的答案（下面 `bc_target`）。
- 抗抖动直接写进策略网络的学习目标：`training.policy_smooth_coef`（主配置 1.0，另外几份都继承）。策略网络的损失再多一项：`policy_smooth_coef` ×（策略网络在这一步和下一步给出的均值动作之差）的平方，批里所有的步都算，不随日程撤（做法出自 CAPS 的时间平滑项，Mysore 2021）。这一项不经过价值网络（为什么要它见 `design.md` 第 9 节）。奖励里的 `action_change` 留着（规划器靠它）。1.0 是第一次试的数。`updates.jsonl` 多 `policy_smooth_loss`。默认 0 = 没有这一项。
- 让更新跑得快：`training.compile_update`（默认 `false`；`guidewire_rl/model/fast_update.py`）。`true` 时一次更新被重排成两个只在显卡上算的损失函数，再由 `torch.compile` 编译成几个大运算；反向、裁剪、Adam、目标网络的软更新照旧。数学上是同一个更新，但浮点舍入和随机数的取法不同，结果是统计意义上相同、不是逐位相同。机器上要有 `gcc`（`scripts/remote/setup_machine.sh` 会装）；每个进程头一分钟在编译；编译失败时打印一次警告 `training.compile_update: compilation failed` 并退回不编译，不会崩。存档不变，开着关着可以互相续跑。`"eager"` 是只重排不编译，和原来的更新结果一致，测试用（`tests/test_fast_update.py`）。
- 考试和训练同时跑：`evaluation.background`（默认 `false` = 训练停下来等考试；只有 `training.async_envs: true` 的训练支持，别的开法报错）。`true` 时到了考试的步数，把当时的策略复制一份，另起一组仿真去考，训练不停。种子、局数、每局的随机数都和停下来考一样，成绩就是那一步的策略的成绩。变的是成绩晚到：`eval.jsonl` 的 `step` 是被考的那一步；日程（撤老师、守门）和 `best.pt` 在成绩到的时候才反应；`events.jsonl` 的 `evaluation` 事件多 `evaluated_step` 和 `background`。`best.pt` 存的是被考的那一步的策略。一次只考一场，上一场没完时最多再排一场（更新的顶掉在等的，记 `evaluation_dropped`）；训练正常结束前会把没考完的考完；被中断时没考完的丢掉、记 `evaluation_dropped`，续跑不补考。考试的仿真和训练的仿真共用 CPU。
- 一场考试最长多久：`evaluation.max_minutes`（默认 0 = 不限）。到点还没结束的局记成仿真错误，照常出成绩，记事件 `evaluation_cut`（被切的种子）。停下来考和同时跑两种都生效。
- 考试里停住的局提前结束：`evaluation.end_when_stalled`（默认 `false`）。`true` 时考试的每一局开局前告诉环境“这一局停住就结束”，由 `env.truncation` 里的 `Stalled` 判（窗口和阈值在那里；环境里没有这个零件时不起作用）。这样结束的局算没成功，结束原因是 `stalled`（成绩里 `end_reasons` 单独一项），不算仿真错误。定期考试（停下来考、同时跑）和 `scripts/evaluate.py` 都认这个键；训练局不受它影响（训练局要开，把 `Stalled` 的 `always` 改成 `true`）。开着它成功率略偏低（窗口 200 步时误判不到 1 % 的成功局），原因和回测的数见 `design.md` 第 8 节；出结论的达标考试不开。
- `scripts/evaluate.py` 开多个进程时：每个进程跑完自己的局就退出；没给 `--device`、且由策略网络出动作（`planning.mpc: false`）时默认在 CPU 上算（给了 `--device` 以它为准）。显卡和 CPU 算出来的动作有舍入差，同一批种子逐局对比时两边要用同一种。
- 不跑仿真、只在回放上学：`scripts/offline_updates.py --config … --checkpoint <存档> --updates N --out <新存档>` 在存档自己的回放上照常更新 N 次，写出能 `--resume` 的存档。它要一个存档作起点；从第 0 步干净重开用下面的 `training.pretrain_updates`。
- 谁出动作：`--set planning.mpc=false`（或配置里 `planning: {mpc: false}`）让策略网络出动作，考试也是；去掉就是规划器出动作。现在用哪个、为什么，见现状页和 `design.md` 第 9 节。
- 影子老师（DAgger 式标签，2026-11-13 晚，配置 `configs/two_devices_ica_dagger.yaml`）：新键都默认关，不写时训练和以前逐位相同。
  - `env.shadow_expert: {type: StagedController, …}`（不写 / `null` / `false` = 关；主配置写 `false`，这样 `--set 'env.shadow_expert={type: StagedController}'` 才能打开）：策略开车时手写推法在环境进程里跟着跑（`env/two_device/shadow.py`），每局 reset 后重置，每步在动作执行**之前**按仿真真值算出它会怎么做，不执行，只放进 `info["expert_action"]`（和回放里存的动作同一个归一化单位），另有 `expert_stage`、`expert_substage`、`expert_off`（没标签的原因：`gave_up:…`、`done`、`error:…`）、`expert_events`（这一步推法日志里新加的条目）。推法认输、做完或报错的步标签是 NaN。手写推法自己开车时（前缀、录示范）标签就是执行的动作；前缀期间影子跟着前缀的推法一步步走，交接时不重置。注意（2026-11-13 在 16 万步策略上量的，`runs/round2_twocrutch_v2/shadow_160k_pi`）：推法的阶段只在策略照它的路走时才往前走，否则停在某一阶段，一直按那一阶段给标签，例如导丝已在颈总里还给“导丝全速往回收”。所以加了 `sync`（`{type: StagedController, sync: true}`，默认关，影子老师配置里开）：每步给标签前拿裁判的阶段（P0–P4）和推法的阶段比，推法落后就只往前挪到这个阶段该在的地方，按推法自己换阶段的方式进入（子阶段、计数、步数预算从进入时算）：P1 → `place:retract`；P2 → 推法自己读到导管头进头臂干 ≥ 2 mm 就 `seat`，否则 `turn`；P3 → 导丝还在导管里（d < 0）且推法读到导管在头臂干里就 `seat`，否则 `wire:out`；P4 → `ica`（目标在颈总时 `wire:up`）。推法已经在这个阶段或更前面就不动。挪完、出完动作以后推法的阶段还落后于这个阶段起码该到的地方（P1 place、P2 turn、P3 seat、P4 wire），这一步没有标签（`expert_off` 为 `behind:…`）。每次挪动记在推法日志里（`sync:旧>新`）。
  - 回放每步多存一列 `expert`（T × 4，NaN = 没标签），随批次抽出（`Batch.expert`）、随存档保存；旧存档和旧示范文件读进来时，前缀步、示范步取动作本身，其余 NaN。`scripts/record_demos.py` 写 `expert_action`（= 动作）；`scripts/evaluate.py --save-episodes` 在帧的 extras 里存上面那几项。
  - `training.bc_target`：`action`（默认，和以前一样：只模仿前缀步、示范步上的动作）或 `expert`（在每个有有限标签的步上模仿 `expert`，策略自己的步也算）。`bc_coef`、随示范配额撤的 `bc_scale`、`bc_loss`/`bc_share` 照旧。
  - `training.self_bc_coef`（默认 0 = 关）：策略再多模仿一样东西——它自己成功、还留在回放最近 `self_demos.capacity` 局里的那些局的动作。误差算法同 `bc_loss`，不随日程撤（老师撤完以后照样在）。回放出批次时把这些步的来源标成 3（`SOURCE_SELF_DEMO`），存着的局不改。更新记录里多 `self_bc_loss`、`self_bc_share`（这一批里这种步占多少）。提速版更新（`compile_update`）里也有，有测试保证两边逐项相同。
  - `training.bc_loss`：模仿的误差怎么算。`l2`（默认）是平方；`smooth_l1` 是绝对值（差在 `training.bc_loss_beta`，默认 0.05，以内仍按平方）。绝对值时小幅度的答案和大幅度的答案拉力一样大，答案冲突时落在多数那一边而不是平均。更新记录里的 `bc_abs_error` 是平均绝对误差，两种算法下都报、可以互相比；`bc_loss` 的数值在两种算法之间不能比。
  - 策略网络的输出（2026-11-13 晚，三项都默认关）：`model.log_std_max` 是策略标准差（tanh 之前）的上限，配置里的值说了算，读旧存档时也按配置改过来（存档里存着这个上限）；影子老师配置写 −1.6（标准差 ≤ 0.2；原来 2，16 万步时标准差就顶在 e² ≈ 7.4，训练里执行的动作在手写推法的局面上近乎随机 ±1）。`training.policy_mean_bound: b` 和 `policy_mean_bound_coef: c`：策略均值在 tanh 之前的软上限，损失加 c × mean(relu(|μ| − b)²)（梯度不经过 tanh；16 万步时 |μ| ≈ 20，tanh 上的损失在那里没有梯度），`updates.jsonl` 多 `policy_mean_over_bound`（超过 b 的比例）、`policy_mean_abs_max`、`policy_mean_bound_loss`；设了 b 时模仿的目标截到 ±tanh(b) 而不是 ±1（满速的目标不再要无穷大的 μ）。影子老师配置 b = 2.5、c = 0.01。这一项按 Adam 每次更新约挪 lr，能防住新开的训练饱和，要把已经饱和的旧存档拉回来得上万次更新。
  - `shadow_expert` 里还可以给手写推法的参数，让它的答案是策略看得出来的（2026-11-14 加，不写就是原来的行为；为什么见 `design.md` 第 10 节）：`first_place: false` 收丝后直接在原地把钩转朝上；`turn_dir: -1` 钩离目标远时永远朝同一个方向转；`turn_timeout` 转钩的时限；`stall_needs_push: true` “推了没动就退一下”只数真推了的步；`give_up: false` 旁观时不认输。每个参数的细节在 `configs/two_devices_ica_dagger.yaml` 开头的注释和 `script.py` 里。
  - `training.pretrain_updates: N`（默认 0）：新开的训练在第一步之前先在回放（示范）上照常更新 N 次，记事件 `pretrain`；回放抽不出样本就报错。这之后智能体算“已训过”，`initial_updates` 那一阵不再跑。配 `training.seed_steps: 0` 时没有随机动作期，第一步就由策略出动作（不配 `pretrain_updates` 而 `seed_steps: 0`、回放里有示范时：单环境和异步并行在第一步之后、同步并行在第一轮 reset 之后立刻跑 `initial_updates`，那第一步 / 第一轮动作出自没训过的策略）。
- 后三种的起点课程是同一个零件（`train/curriculum.py` 的 `StartCurriculum`），差别只在 `training.start_curriculum.supply`。`self` 和 `demos` 都从起点库（`train/start_library.py`）抽存档恢复开局，课程规则相同：30 % 的局从插入点起，其余练前沿格，这一格成功率过半前沿就往插入点挪一格。从存档起的局按**抽中的那一格**记账（= 存档存下时所在的格、库里它所在的文件夹），不按开局时重新量的进度换算；开局进度照旧记在 `start_ratio`（2026-11-13 改：存档压在格的下边界上，换个目标重新量会差约 0.01，以前约一半被记到低一格）。步数上限：`env.start` 的 `count_saved_steps: true`（“示范当起点”）时从存档起的局扣掉存档前已走的步数（存在第 419 步的存档开局后剩 381 步），默认 `false`（“自己的局当起点”）重新给 800 步。
- `supply: demos` 的库是固定库：开训时读 `library:` 指的目录（相对路径从工作区根目录算），训练中不往里加、不删文件、环境不沿途存档，所以几次训练可以共用一份。目录里没有哪一格够 `library_min` 份时开训直接报错。
- 造“示范当起点”要用的示范和库（一次，约 20 局手写推法；换了观测或奖励原料的格式要重造）：

```bash
conda run --no-capture-output -n guidewire_rl python scripts/record_demos.py \
    --config configs/two_devices_ica_demostart.yaml --seeds 1000-1019 --workers 20 --episode-timeout 1800 \
    --out runs/demos_round2c --starts runs/demos_round2c/starts --starts-from-failed
```

  一局手写推法约 500 步，机器忙时一局要十几分钟，默认每局 600 秒的期限不够，所以给 1800 秒；20 个进程一起录约 11–12 分钟。

  现在磁盘上的示范：`runs/demos_round2c`（种子 1000–1019，20 局里 17 局成功，另 3 局是手写推法认输的半截局；起点库 191 份；主配置和“示范当起点”用），`runs/demos_round2e`（种子 2000–2059 里成功的局，用 2026-11-14 改过的手写推法录；老师旁观的配置用）。更早的几批已经删了，各批录制的经过在工作日志。

  `--starts` 让录制时每局第一次走进一个新的 10 % 格就存一份状态，默认只留成功局的（`start_library.collect_episode_states`），加 `--starts-from-failed` 时认输或步数到顶的局的也留（用户 2026-11-13：“也放进开局池”；存档是往前走、第一次进一格时存的，不是出事那一步），放在 `<目录>/grid_KK/demo_<种子>_s<步>.npz`。
- “示范当起点”用和“两根拐棍”同一份日程（`train/schedule.py`）：示范保底配额 25 %（手写示范和策略自己的成功局同池），前沿挪回插入点、最近两次评估合起来成功率 ≥ 50 % 后 15 万步撤到 0，带守门；收尾 5 次评估不创新高停训。日程全套最快约 52.5 万步，30 万步时轮不到撤示范。两份配置的日程逐项相同：梯子关、受力扣分先轻后重（上一节日程那条）。它也带模仿项（`bc_coef` 从主配置继承）。
- 短跑试车（改了接线以后先这样试一下）：2 个进程、240 步、关 wandb。一局 800 步的配置在 240 步里一局都结束不了、回放是空的、不会更新，要看更新就加 `--set 'env.truncation=[{type: MaxSteps, max_steps: 40}, SimError, WallBreach, TipForceBlowup]'`；“两根拐棍”课程开局在第 10 格，手写推法约 425 步才走到，`prefix_max_steps` 小于这个数只会看到 `out_of_steps`（就地接手），看不到 `handover`；前缀步算进总步数。
- 想让“自己的局当起点”接着某次训练的库：`--resume` 那次的存档并用同一个 `--run-name`（库从 `runs/<run>/starts/` 读回）。

## 还没定的

- 受力损失的写法：G13 用的是 Huber（smooth L1，β = 1，即配置里的 `huber`）。依据有两条：Wilson 的训练分支（`reference/contact-aware-tdmpc2-training/`）这样写；G13 存档里优化器存的受力头梯度大小也只和 Huber 对得上（2026-10-25 核过）。`configs/g13.yaml` 里现在还是 `mse`，改不改等用户定。
- 新环境的求解器（上游 LCP 还是 G13 那种）和导丝参数，见 `simulator_changes.md` 的待定事项。
