# 训练框架怎么用

代码在 `guidewire_rl/`，一个实验 = `configs/` 里一份 YAML。

## 两个环境

| conda 环境 | 仿真器 | 配置 | 用来 |
| --- | --- | --- | --- |
| `guidewire_rl` | SOFA 25.12 + 工作区 `stEVE/`（上游）+ 我们的修正 | `configs/g13_upstream.yaml`（合成血管）、`configs/phantom_g13.yaml`（lab 的 phantom）、`configs/phantom_friction.yaml`（phantom，摩擦 0.1）、`configs/phantom_root.yaml`（再把主动脉中心线延到根部的底）、`configs/phantom_all_vessels.yaml`（再给 phantom 每条血管都配中心线）、`configs/phantom_real_mass.yaml`（phantom，导丝真实质量 + 2 ms 物理步 + 碰撞预判 2 mm）、`configs/two_devices_real_mass.yaml`（导丝 + 钩形导管，两根都按真实质量，每根器械按自己的半径碰墙） | **新的训练都用这个** |
| `contact_aware_tdmpc2` | SOFA 23.06 + Wilson 自带的 eve | `configs/g13.yaml` | 复现、对照 G13 |

两个环境装的是同一份 `guidewire_rl` 代码。`configs/g13_upstream.yaml` 继承 `g13.yaml`，只把导丝换成上游的 Angled。`configs/phantom_g13.yaml` 再继承它，只换两处：血管换成 lab 的 phantom（`BiroPhantomArch`，数据和已知问题见 `guidewire_rl/env/vessels/biro_phantom/README.md`），血管几何量改用 `frame: vessel`。`configs/phantom_friction.yaml` 继承 `phantom_g13.yaml`，只把导丝和管壁之间的摩擦系数从 0.001 改成 0.1（2026-10-27 用户决定，来历见该文件注释）。`configs/phantom_root.yaml` 继承 `phantom_friction.yaml`，只把血管换成主动脉中心线延到根部底的版本（`BiroPhantomArch(centerline: root_extended)`，碰撞网格不变；为什么和怎么做见 `biro_phantom/README.md`）。phantom 的血管还可以按整套版本选：`vessel_tree: {type: BiroPhantomArch, version: rebuilt_cut | spheres_cut | lab_cut | rebuilt | spheres | lab}`（`rebuilt*` 模型内壁版、`spheres*` 内接球版，都只用 STL 生成；`lab*` 是 lab 输入做的内接球版，留着对照；`_cut` 只有头和胸腔 / 不带后缀是完整带右臂；是什么、怎么来的见 `guidewire_rl/env/vessels/biro_phantom/README.md`，2026-10-28/29 做的，还没有配置用它们）。`configs/phantom_all_vessels.yaml` 继承 `phantom_root.yaml`，只把血管换成每条血管都有中心线的版本（`centerline: all_vessels`）：目标点和 2D 坐标范围仍只按原来 6 条，离壁距离、奖励里的路程、stEVE 停推的位置看全部 75 条。`configs/phantom_real_mass.yaml` 继承 `phantom_friction.yaml`，把导丝密度改成镍钛的真实值（stEVE 的数是单位写错，重约 3300 倍）、物理步 6 → 2 ms、碰撞预判范围 0.5 → 2 mm、受力标签每控制步 66 个小步（2026-10-28 用户决定；为什么必须一起改、代价多少见 `simulator_changes.md` 问题 20）。`configs/two_devices_real_mass.yaml` 继承 `two_devices_round1.yaml`，两根器械改成真实质量（导丝同上，导管 1.2e-9 是按塑料约 1.2 g/cm³ 估的）、2 ms、预判 2 mm，导管保留钩形、刚度换成真实导管的（E 4000 → 660 MPa），仿真器打开四个开关：每根器械一份碰撞点、按自己的外半径碰墙（`instrument_collision`，问题 8）、最短梁 0.05 mm（`min_beam_length`，问题 22）、两根重叠处按更密的那根排节点（`densest_sampling`，问题 23）、某个小步出现 NaN 就停下这一步（`stop_on_nan`）；最短梁和按密的排节点要我们编的 BeamAdapter。两个受力标签每控制步 66 个小步、读各自器械的碰撞点（2026-11-03 用户决定）。两套仿真器在物理上的差别，以及我们改了仿真器的哪些地方，见 [`simulator_changes.md`](simulator_changes.md)。

## 结构

| 位置 | 管什么 | 来源 |
| --- | --- | --- |
| `guidewire_rl/env/` | 按 YAML 的 `env:` 一节拼环境：仿真场景、观测、奖励、终止/截断、label | stEVE 自带零件 + 我们两个：`VesselGeometry`（G13 的 8 个几何量）、`DistalForce`（尖端接触力） |
| `simulator_modifications/`（工作区根目录） | 我们对仿真器的所有修改：导管空心、带种子重置（用 stEVE 原类的名字登记，只在上游 stEVE 下生效），以及 BeamAdapter 的补丁和编译脚本 | 见那里的 README 和 `simulator_changes.md` |
| `guidewire_rl/model/` | 世界模型、规划、训练步骤、额外预测头 | 世界模型和规划从参考包原样复制；额外预测头是新加的 |
| `guidewire_rl/train/` | 回放、训练主循环（一个环境 `trainer.py`；多个环境并行 `parallel_trainer.py` + `env_pool.py`）、存档、评估 | 照官方 TD-MPC2 的写法；并行是我们加的 |
| `scripts/` | `train.py` 训练、`evaluate.py` 评估、`inspect_env.py` 看观测 | — |
| `runs/<名字>/` | 训练产物：`config.yaml`、`*.jsonl` 日志、`checkpoints/` | — |

## 改东西

改 YAML，不改代码。观测维度、模型输入层会自动跟着变；改了观测就不能接着旧 checkpoint 训。

- **观测**：`env.observation` 每行一项，按顺序拼起来。删一行、加一行，或改 `VesselGeometry` 的 `features`（`clearance` `heading` `bend` `radius` `valid` 任选）。`frame: reference` 是 G13 原样，`frame: vessel` 是修正坐标系后的离壁距离。
- **奖励 / 终止 / 截断**：`env.reward` 等，每行一个 stEVE 零件，多行自动相加或合并。
- **一局怎么结束**（两器械）：成功和硬失败是真结束（裁判判，`terminated`，价值不往后估）；超时 `MaxSteps`、数值崩 `SimError`、穿墙 `WallBreach` 是截断。穿墙 = 某根器械的中心线出了血管网格，按仿真器的故障处理（2026-11-09 用户定）：`WallInfo` 把它写进 `info["wall_breached"]`，训练器把那一步丢掉、这局按 `breach` 收尾（`episodes.jsonl` 的 `end`，`metrics.jsonl` 的 `breach_rate` 和 `wall_breaches`，评估报告的 `end_reasons`），不给 −1，和数值崩分开计数；"别顶墙"由受力扣分教。裁判的"头出中心线管子 6 mm"硬失败留着当保底，正常情况下穿墙先触发。
- **label**：`env.labels`，仿真每步量的、控制器看不到的量，只当训练答案。
- **额外预测头**：`aux_heads`，每个头指定答案用哪个 label、损失写法、权重，以及 `gradient`（`shared` 会影响模型，`detached` 只在旁边学）。
- **新实验**：新建一份 YAML，第一行写 `extends: g13.yaml`，只写要改的部分。字典逐项合并，列表整个替换；写 `null` 就是删掉继承来的那一项，比如换一种血管时，用 `vessel_tree: {type: BiroPhantomArch, seed: null, scaling_xyzd: null}` 去掉 `AorticArch` 的参数。
- **临时改一个数**：命令行加 `--set 段.键=值`，例如 `--set training.total_steps=1000`。
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
# 测试：全部（约 1–2 分钟，含真实 SOFA 场景）。两个环境都要跑：
#   guidewire_rl 跑仿真器修正（upstream），跳过和参考包对照的（reference）；
#   contact_aware_tdmpc2 正好相反。
conda run --no-capture-output -n guidewire_rl python -m pytest -q
conda run --no-capture-output -n contact_aware_tdmpc2 python -m pytest -q
# 训练
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/g13_upstream.yaml
# 断了接着训
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/g13_upstream.yaml --resume runs/<名字>/checkpoints/step_N.pt
# 从 G13 的 checkpoint 接着训（连它的回放一起）。G13 的回放来自旧仿真器，所以这条只在旧环境用过：
conda run --no-capture-output -n contact_aware_tdmpc2 python scripts/train.py --config configs/g13.yaml --init-from-reference reference/checkpoints_g13/step_260000.pt
# 评估
conda run --no-capture-output -n guidewire_rl python scripts/evaluate.py --config configs/g13_upstream.yaml --checkpoint <checkpoint>
```

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

## 还没定的

- 受力损失的写法：G13 用的是 Huber（smooth L1，β = 1，即配置里的 `huber`）。依据有两条：Wilson 的训练分支（`reference/contact-aware-tdmpc2-training/`）这样写；G13 存档里优化器存的受力头梯度大小也只和 Huber 对得上（2026-10-25 核过）。`configs/g13.yaml` 里现在还是 `mse`，改不改等用户定。
- 新环境的求解器（上游 LCP 还是 G13 那种）和导丝参数，见 `simulator_changes.md` 的待定事项。
