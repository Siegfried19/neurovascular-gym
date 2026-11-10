# neurovascular-gym

在 BIRO 实验室的神经血管 phantom（头颈血管模型）里，用强化学习训练"导丝 + 导管"从降主动脉走到右颈内动脉。仿真器是 SOFA + BeamAdapter（打了我们的补丁）+ stEVE（上游，一行没改）+ 本仓库对它们的修正；算法是 TD-MPC2（世界模型 + 隐空间规划），世界模型和规划器从 Wilson Wan 的 contact-aware TD-MPC2 复制过来。一个实验 = `configs/` 里一份 YAML。

这份 README 是给人和给代码代理（Claude Code、Codex）一起看的：路径、命令、不能动的东西都写死。读完它再动手。

## 仓库里有什么

| 位置 | 是什么 |
| --- | --- |
| `guidewire_rl/` | 我们的代码，一个 Python 包：`env/`（MDP 一侧）、`model/`（求解一侧）、`train/`（训练循环）、`config.py`（读 YAML） |
| `simulator_modifications/` | 对仿真器的修改：7 个 Python 文件用 stEVE 原类名登记修正版，`beamadapter/` 是 BeamAdapter 的 6 个补丁和编译脚本 |
| `configs/` | 10 份 YAML，链式继承（见「配置」） |
| `scripts/` | 入口：`train.py`、`evaluate.py`、`inspect_env.py`、`record_demos.py`、`make_snapshots.py` |
| `tests/` | pytest，一个零件大致一个文件（见「测试」） |
| `visual/workbench/` | viser 工作台：选 phantom 版本和设置、部署策略、看器械和中心线、X 光模式。浏览器里看，Python 端跑仿真 |
| `docs/` | 三份技术文档：`training_framework.md`（框架怎么用）、`simulator_changes.md`（仿真器每个问题的来龙去脉，按问题编号）、`simulation_speed.md`（提速这条线） |
| `environment.yml`、`pyproject.toml` | conda 环境和包定义 |
| `THIRD_PARTY_NOTICES.md` | 搬来的代码和数据各自从哪来、什么授权 |

不在仓库里、要自己装的：SOFA 25.12、stEVE（固定提交）、我们编的 BeamAdapter。可选的：`reference/`（Wilson 的原版包和 G13 checkpoint，只有复现 G13 时要）。产物目录 `runs/`、`outputs/` 不进仓库。

各文件夹里自己的 `README.md`（`guidewire_rl/env/two_device/`、`env/c_arm/`、`env/vessels/biro_phantom/`、`env/instruments/data/`、`simulator_modifications/`、`visual/workbench/`）讲那个文件夹里每个文件干什么、数据从哪来。它们和 `docs/` 里有些链接指向内部的计划和日志（`todo_list/`、`agent_log/`），那些不在本仓库里，看到就跳过。

## 安装（Linux，2026-11-10 实际在用的版本）

路径按我们机器写（`~/opt/…`、`~/Project/…`）；换地方要同步改环境变量和 `build.sh` 开头的变量。

1. **SOFA v25.12.00 的 Linux 二进制包**，解压到 `~/opt/SOFA_v25.12.00_Linux`（从 `sofa-framework/sofa` 的 GitHub release 下载）。它自带 SofaPython3（Python 3.12）和 BeamAdapter；自带的 BeamAdapter 我们不用（见第 5 步）。

2. **conda 环境**：
   ```bash
   conda env create -f environment.yml          # 环境名 guidewire_rl，Python 3.12，torch 是 CUDA 12.8 的轮子
   ```

3. **stEVE**（包名 `eve`），克隆到本仓库根目录的 `stEVE/`（`.gitignore` 已排除），钉在上游提交 `c3704ccf8f5f6deca443ca98f51a023c2f2b4838`（分支 `feat/sofa25.12_compatibility`，2026-04-14），然后两个 editable 安装：
   ```bash
   git clone https://github.com/lkarstensen/stEVE.git stEVE
   git -C stEVE checkout c3704ccf8f5f6deca443ca98f51a023c2f2b4838
   conda run -n guidewire_rl python -m pip install -e stEVE
   conda run -n guidewire_rl python -m pip install -e .        # 装 guidewire_rl 和 simulator_modifications
   ```

4. **SOFA 的环境变量**写进 conda 环境（`conda run` 和 `conda activate` 都会带上）。顺序有讲究：`SOFA_PLUGIN_PATH` 第一项必须是我们编的 BeamAdapter，`LD_LIBRARY_PATH` 末尾必须带本环境的 `lib`（SofaPython3 要在那里找 `libpython3.12.so.1.0`）：
   ```bash
   SOFA=$HOME/opt/SOFA_v25.12.00_Linux
   ENV=$(conda run -n guidewire_rl python -c 'import sys; print(sys.prefix)')
   conda env config vars set -n guidewire_rl \
     SOFA_ROOT=$SOFA \
     SOFA_PLUGIN_PATH=$HOME/opt/BeamAdapter-v25.12-dev:$SOFA/plugins \
     PYTHONPATH=$SOFA/plugins/SofaPython3/lib/python3/site-packages \
     LD_LIBRARY_PATH=$SOFA/lib:$SOFA/plugins/SofaPython3/lib:$ENV/lib
   ```

5. **编我们的 BeamAdapter**：
   ```bash
   bash simulator_modifications/beamadapter/build.sh
   ```
   它把上游 BeamAdapter 在 SOFA 25.12 自带的那个提交检出到 `simulator_modifications/beamadapter/build/`，按顺序打 6 个补丁，编译后装到 `~/opt/BeamAdapter-v25.12-dev/BeamAdapter`。需要：系统 g++ 11、cmake、ninja、Boost 头文件（脚本从环境变量 `TOOLS` 指的目录找 `bin/cmake`、`bin/ninja`、`include/`，默认 `~/anaconda3/envs/sofa-build`，自己建一个装了 `cmake ninja boost-cpp` 的 conda 环境即可）；Eigen 3.3.7 脚本自己下载。为什么必须编：SOFA 自带的 BeamAdapter 在器械推进正好 0.01 mm 时段错误（官方后来修了，补丁之一），另外我们的四个补丁给了配置里用到的开关（最短梁、重叠处按更密的器械排节点、套在导管里的导丝不放碰撞点、每器械碰撞点的即时重排）。`build/` 是中间物，可以随时删。

6. **跑任何 Python 都用**
   ```bash
   conda run --no-capture-output -n guidewire_rl python …
   ```
   直接调环境目录里的 `python` 拿不到第 4 步的变量，`import Sofa` 会失败。

### 验证装好了

```bash
# 不起 SOFA 的测试，几分钟
conda run --no-capture-output -n guidewire_rl python -m pytest -q -m "not sofa and not checkpoint"
# 全部，含真实 SOFA 场景（手写推法整局等），约 20–30 分钟
conda run --no-capture-output -n guidewire_rl python -m pytest -q
```

预期：没有失败；标了 `checkpoint` / `reference` 的几项因为没有 `reference/` 自动跳过，标 `slow` 的（C 臂选角搜索，几分钟一项）要加 `--run-slow` 才跑。标记的定义在 `pyproject.toml`，跳过逻辑在 `tests/conftest.py`。

### 可选：`reference/` 和旧环境

`guidewire_rl/__init__.py` 里 `REFERENCE_ROOT = <仓库根>/reference`，`REFERENCE_G13_CHECKPOINT = reference/checkpoints_g13/step_260000.pt`。放上 Wilson 的 `contact-aware-tdmpc2` 仓库和 G13 的 checkpoint，`tests/test_reference_fidelity.py` 和 `configs/g13.yaml` 的逐位对照才会跑；`scripts/train.py --init-from-reference` 也读它。精确复现 G13 还要第二个 conda 环境 `contact_aware_tdmpc2`（Python 3.10、SOFA 23.06、`eve` 从 Wilson 的包导入），见 `docs/training_framework.md`「两个环境」。没有这些，新训练一点不受影响。

## 怎么跑

```bash
# 训练（产物 runs/<run.name>/：config.yaml、episodes.jsonl、metrics.jsonl、checkpoints/）
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml \
    --set training.total_steps=1000 --run-name smoke          # 临时改一个数、换个名字
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml \
    --resume runs/<run>/checkpoints/step_N.pt
# 评估一个 checkpoint（固定种子、规划器取均值动作）
conda run --no-capture-output -n guidewire_rl python scripts/evaluate.py --config configs/two_devices_ica.yaml \
    --checkpoint runs/<run>/checkpoints/step_N.pt --episodes 20 --workers 8
# 看策略会看到什么（观测布局、几步随机动作的奖励和标签）
conda run --no-capture-output -n guidewire_rl python scripts/inspect_env.py --config configs/two_devices_ica.yaml
# 手写推法录示范（给训练器当 demos）
conda run --no-capture-output -n guidewire_rl python scripts/record_demos.py --config configs/two_devices_ica.yaml \
    --seeds 1000-1019 --workers 10 --out runs/demos_ica
# 开局存档库（手写推法走到各阶段的仿真状态，供 FromSnapshot 开局）
conda run --no-capture-output -n guidewire_rl python scripts/make_snapshots.py --help
# 工作台（之后用 Chrome 开 http://localhost:8080，要 WebGL）
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py --env two_devices_ica
```

每个脚本开头的文档字符串写了全部参数。

## 地图：每个文件夹回答什么问题

### `guidewire_rl/` 按强化学习的三块分

```
guidewire_rl/
├─ env/      MDP 一侧：仿真场景、观测、动作、奖励、一局怎么开始和结束、标签（训练答案，策略看不到）
├─ model/    求解一侧：世界模型、规划器、更新步骤（agent.py、world_model.py）、附加预测头（aux_heads.py）
├─ train/    接起来跑：回放（replay）、单环境循环（trainer）、多进程（parallel_trainer + env_pool）、
│            课程（curriculum）、示范（demos）、存档（checkpoint）、评估（evaluate）、指标（metrics）、日志（logger）、组装（setup）
└─ config.py 读 YAML：extends 继承 + --set 覆盖
```

YAML 的 `env:` 一节对应 `env/`，`model:` 和 `training:` 对应另外两个。

### `env/` 里是两套分法混着

- **按零件种类**：`observations/`（观测项）、`labels/`（标签）、`instruments/`（器械）、`vessels/`（血管）、`c_arm/`（成像角度）。每个文件夹一类可换零件，配置里按名字选。顶层 `builder.py`（拼场景）、`registry.py`（名字 → 类）、`guidewire_env.py`（gymnasium 包装）、`actions.py`（动作 → stEVE 指令）是拼装框架。
- **按任务**：`two_device/` 把两器械任务的所有零件放在一起：`route.py`（路线和尺子）、`phases.py`（裁判：五段、成功、硬失败）、`observations.py`、`rewards.py`、`targets.py`、`start.py`、`terminal.py`、`truncation.py`、`wall.py`（离墙、穿墙）、`labels.py`、`actions.py`（安全壳）、`snapshot.py`（开局存档恢复）、`script.py`（手写推法，训练的老师，不进配置）。零件表在那里的 `README.md`。
- `vessels/` 里三种东西混着：运行时把血管接进 stEVE 的类（`biro_phantom.py`）、离线从 STL 生成网格和中心线的脚本（现在用的 `biro_rebuild.py`、`biro_spheres.py`、`biro_cut.py`；早期版本 `biro_phantom_root.py`、`biro_phantom_all.py`、`biro_phantom_lumen*.py`）、数据目录 `biro_phantom/`（每份文件是什么、哈希、怎么重新生成：那里的 `README.md`）。

### 名字怎么变成类

配置里写 `type: SofaBeamAdapter`、`type: BiroPhantomArch`、`type: PhaseDetector` 这样的名字，`env/registry.py` 查表拿到类：stEVE 自带的零件按 stEVE 的名字，我们的零件用 `@register("observation")` 之类登记。`simulator_modifications/` 用 **stEVE 原来的类名**登记修正版（`SofaBeamAdapter`、`CenterlineRandom`、`Manual`、`Tracking2D`、`NormalizeTracking2DEpisode`、`BruteForceBFS`、`TrackingOnly`），所以配置照常写原名，拿到的就是修过的；只在上游 stEVE 下生效（`simulator_modifications.UPSTREAM`），Wilson 的旧 eve 下不登记。

### 配置

链式继承，字典逐项合并、列表整个替换、写 `null` 删掉继承来的项：

```
g13.yaml                      G13 复现：每个数来自 G13 checkpoint 里存的训练配置（Wilson 的 eve，旧环境）
└─ g13_upstream.yaml          同一任务搬到上游 stEVE（导丝换成上游的 Angled）
   └─ phantom_g13.yaml        血管换成 phantom
      └─ phantom_friction.yaml      摩擦 0.001 → 0.1
         ├─ phantom_root.yaml → phantom_all_vessels.yaml   主动脉中心线延到根部 / 每条血管都有中心线（lab 输入版）
         ├─ phantom_real_mass.yaml   单导丝，真实质量、2 ms 物理步
         └─ two_devices_round1.yaml  导丝 + 钩形导管，目标右颈总（第一轮）
            └─ two_devices_real_mass.yaml   两根真实质量、4 ms 物理步、仿真器开关全开
               └─ two_devices_ica.yaml      目标右颈内，C 臂按目标选角，γ 钉死 —— 现在用的
```

每份文件的头注释写了相对上一级改了什么、为什么、哪天定的。`g13.yaml` 只在旧环境下能跑。

### 数据怎么流

```
configs/*.yaml → scripts/train.py → config.py
  → env/builder.py 按 registry 找零件 → stEVE 场景（simulator_modifications 垫在下面）→ GuidewireEnv
  → train/trainer 跑循环，调 model/agent 规划和更新
  → runs/<名字>/
```

## 改东西去哪

- **加观测 / 奖励 / 标签 / 终止 / 截断**：写一个 stEVE 对应基类的子类，`@register("observation")` 等登记，YAML 里按名字加一行。观测维度和模型输入层自动跟着变；改了观测就不能接旧 checkpoint 训。
- **改仿真器行为**（接触、碰撞、速度、种子、坐标换算）：`simulator_modifications/`。Python 能解决的写成子类按原名登记；要动 BeamAdapter 的 C++ 就加补丁：在 `beamadapter/build/src`（已打上现有补丁并 `git add -A` 的基线）上改，`git -C build/src diff > 新名字.patch`，列进 `build.sh` 的 `PATCHES`，重编。`build.sh` 每次从锚定提交重置再打全部补丁，手改不存成补丁会丢。每个改动在 `docs/simulator_changes.md` 记一个问题编号。
- **换血管数据**：不手改数据文件。重新生成（命令在 `env/vessels/biro_phantom/README.md`），更新 `tests/test_biro_phantom.py` 里的哈希和 README 的表。
- **换 C 臂角度表**：`env/c_arm/choose_angle.py` 重算，写 `env/c_arm/data/`，见 `env/c_arm/README.md`。
- **新实验**：新建 YAML，第一行 `extends: two_devices_ica.yaml`，只写要改的。临时改一个数用 `--set 段.键=值`。

### 不能手改的文件

- 搬来的代码（和来源保持一致，改动要写明）：`guidewire_rl/model/agent.py`、`model/world_model.py`、`env/labels/force_reconstruction.py`、`env/labels/distal_force.py`、`env/observations/vessel_geometry.py`；BeamAdapter 的两个官方补丁 `official_irc_fix.patch`、`official_rod_straight_fix.patch`。
- 带哈希锁的数据：`env/vessels/biro_phantom/*`（`tests/test_biro_phantom.py`）、`env/instruments/data/simmons.csv`（`tests/test_instruments.py`）、`env/c_arm/data/*.json`、`visual/workbench/assets/makehuman_base.obj`。

## 测试

`pytest -q -m "not sofa and not checkpoint"` 几分钟；全量约 20–30 分钟。改完代码先跑直接覆盖它的文件，提交前跑全量。

| 改了什么 | 跑哪个 |
| --- | --- |
| `config.py` / `registry.py` | `test_config.py`、`test_registry.py` |
| `model/` | `test_agent.py` |
| `train/` | `test_trainer.py`、`test_parallel.py`、`test_replay.py`、`test_two_device_training.py`（假环境） |
| `two_device/` 的零件 | `test_two_device_env.py`；手写推法 `test_two_device_script.py`（含 SOFA 整局）；离墙 / 穿墙 `test_wall_monitor.py`；受力标签 `test_tip_force_fast.py`；力闸 `test_force_blowup.py`；存档 `test_snapshot.py` |
| `simulator_modifications/` | `test_simulator_modifications.py`、`test_device_contact.py`、`test_trim_sheathed.py`（都要 SOFA） |
| `vessels/`、`instruments/`、`c_arm/`、`observations/` | `test_biro_phantom.py`、`test_instruments.py`、`test_c_arm*.py`、`test_vessel_geometry.py` |
| 场景拼装整体 | `test_env_sofa.py` |
| 工作台 | `visual/workbench/tests/test_parts.py` |

两器械的 SOFA 测试共用 `tests/two_device_scenes.py`（建场景、推器械、读碰撞链）。

## 现状（2026-11-10 的快照，之后以提交记录为准）

- **第一轮两器械训练**（2026-10-31 至 11-02，`two_devices_round1.yaml`，170 万步）成功率 0，复盘定为奖励方向的问题。第二轮正在按数据流分块重新设计。
- **已经写进代码和配置的**（`two_devices_ica.yaml`）：目标右颈内；两根真实质量；物理步 4 ms、碰撞预判 1.0 mm、套在导管里的导丝不放碰撞点（三项合计比提速前的 2 ms 配置快约 2.3 倍，验证数在 `docs/simulation_speed.md`；真实质量本身比第一轮的默认质量慢，所以仍慢于第一轮的每步 0.1 s）；穿墙当仿真故障（那一步丢掉、这局按 `breach` 收尾、单独计数）；任一接触点法向力 > 30 N 当仿真错误截断；手写推法会认输；C 臂按目标所在分支查表选角；折扣因子钉死 0.99375。
- **零件做好、还没写进训练配置的**：开局存档恢复（`two_device/snapshot.py` 的 `FromSnapshot`，`scripts/make_snapshots.py` 建库）。
- **设计定了、代码还没改的**（第二轮第 1 块）：安全壳只剩长度限位（现在代码里还是第一轮的壳：限速、力限、同长保护等）；受力进奖励不进壳；路口地图改成这条路线上 6 个路口一次全给；任务框按整棵血管树算；观测从 110 维到 120 维。看到代码和这段不一致，以代码为准、以这段为方向。
- **已知的坑**：SOFA 自带的 BeamAdapter 推进正好 0.01 mm 段错误（所以必须用我们编的）；两根器械同长时 BeamAdapter 刷 "Case 1" 警告，是它自己的 0.01 mm 阈值，无害；`conda run` 不带 `--no-capture-output` 看不到实时输出；仿真出现 NaN（`SimError`）、穿墙（`WallBreach`）、力爆（`TipForceBlowup`）都是截断不是终止，训练器把那一步丢掉；工作台要 Chrome 的 WebGL，桌面端内嵌浏览器画不出来。

## 来源与授权

本仓库的代码 MIT（`LICENSE`）。搬来的代码和数据（Wilson 的 TD-MPC2 实现、实验室的 phantom 数据和导管代码、BeamAdapter 官方补丁、Simmons 导管描迹、MakeHuman 人体网格）各自的来源和授权在 `THIRD_PARTY_NOTICES.md`。
