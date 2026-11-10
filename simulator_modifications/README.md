# 仿真器修改

我们对仿真器（SOFA / BeamAdapter / stEVE）做的所有修改都放在这里。每个问题的来龙去脉、查证过程和决定，见 [`docs/simulator_changes.md`](../docs/simulator_changes.md)，编号和那里一致。

| 文件 | 改的是谁 | 对应问题 | 做了什么 |
| --- | --- | --- | --- |
| `hollow_catheter.py` | stEVE | 问题 2 | stEVE 建器械截面时只传外径，导管被当成实心杆。这个文件在器械建好后给每个截面补上内径（lab 的写法） |
| `speedups.py` | stEVE / SOFA 设置 | 问题 4、5、16、20 | 碰撞树 6 → 16 层；碰撞检测和自由运动并行（2 个线程），phantom 上仿真快约 4 倍；每局重建场景，绕开越跑越慢的问题；碰撞预判范围（`alarm_distance`，stEVE 写死 0.5 mm）做成参数，`phantom_real_mass.yaml` 填 2 mm。这个类继承空心导管修正 |
| `seeds.py` | stEVE | 问题 7、19 | Python 3.12 不接受 numpy 整数做种子，stEVE 带种子重置会报错。这个文件让 3 个类先把种子转成普通整数。固定目标点 `Manual` 另补两处：配置里的坐标列表转成数组，补上 2D 坐标。`CenterlineRandom` 另外：血管树标了 `task_branch_names` 时，只按这几条判断目标点是否落在别的血管里 |
| `observations.py` | stEVE | 问题 14、15 | 观测的两处修正：`Tracking2D` 取点时不再把几个点叠在同一处；`NormalizeTracking2DEpisode` 的 [-1, 1] 范围改成整个血管腔（血管树标了 `task_branch_names` 时只按这几条，见 phantom 的 `all_vessels`） |
| `pathfinder.py` | stEVE | 问题 18 | 奖励里的寻路 `BruteForceBFS` 每步不再深拷贝整张血管图，路径逐位相同 |
| `fluoroscopy.py` | stEVE | 问题 21 | X 光坐标范围转全部 8 个角再取包围盒，C 臂转角大时不再报错（按目标定 C 臂角度时单导丝配置要它）；观测数值不变 |
| `beamadapter/official_irc_fix.patch` | BeamAdapter | 问题 1 | BeamAdapter 官方主分支的推进控制器修复（5 处），修推进正好 0.01 mm 时的越界崩溃 |
| `beamadapter/official_rod_straight_fix.patch` | BeamAdapter | 问题 17 | BeamAdapter 官方的直杆段修复（1 行，v26.06 里有）：不是第一段的直杆段，静止形状多算了一次起点 |
| `device_contact.py` | stEVE 场景设置 / BeamAdapter 参数 | 问题 8、22、23、26 | 五个开关，默认都关（现有配置建出来的场景不变），`two_devices_real_mass.yaml` 开前四个：`instrument_collision` 每根器械一份碰撞点（照 stEVE 显示模型的写法，`AdaptiveBeamMapping` 接这根器械自己的插值），接触距离填它的外半径，几份都在同一个碰撞组、彼此不碰，全局接触距离改 0；`min_beam_length` 最短梁（设 BeamAdapter 补丁加的 `minBeamLength`）；`stop_on_nan` 每个小步查节点，出现 NaN 或飞出 10 m 就像 stEVE 那样停下、标仿真错误，不再卡在下一次碰撞检测；`densest_sampling` 重叠处按更密的器械排节点（设 BeamAdapter 补丁加的 `densestSampling`）。`trim_sheathed` 套在导管里的导丝不放碰撞点（每个小步把导丝碰撞链的起点挪到导管头；设 BeamAdapter 补丁加的 `startCurvAbs`）。这个类继承 `speedups.py` 的那个，`type: SofaBeamAdapter` 解析到它 |
| `beamadapter/min_beam_length.patch` | BeamAdapter | 问题 22 | 我们自己的补丁（4 行）：推进控制器加参数 `minBeamLength`，排节点时离前一个节点比它近的节点合并掉（原来只按 `threshold` 0.01 mm 合并）；`threshold` 的其他用处不变；默认 0 时和不打补丁逐位相同 |
| `beamadapter/adaptive_mapping_refresh.patch` | BeamAdapter | 问题 8 | 我们自己的补丁（两处，约 30 行）：`AdaptiveBeamMapping` 打开监听时，一是在每个小步开头（推进控制器重排梁之后、碰撞检测之前）就按新的梁重排碰撞点，否则每根器械一份碰撞点时碰撞检测用的是上一步的点、接触力会加到别的节点上；二是线段数变了才重建线段并让拓扑版本号加一，否则线段碰撞模型一直是 0 节线段、只有节点在碰。没打开监听的映射不受影响 |
| `beamadapter/densest_sampling.patch` | BeamAdapter | 问题 23 | 我们自己的补丁（80 行）：推进控制器加开关 `densestSampling`，两根器械重叠的地方每一截按覆盖它的各段里最密的间距排节点（原来只按排在前面那根的）；关着时逐位不变 |
| `beamadapter/collision_start_abscissa.patch` | BeamAdapter | 问题 26 | 我们自己的补丁（约 25 行）：`AdaptiveBeamMapping` 加参数 `startCurvAbs`，按每梁点数铺点时跳过这个弧长以前的点（只留它前面最后一个，让跨过起点的那段线段还在）；默认 0 时逐位不变。`device_contact.py` 的 `trim_sheathed` 每个小步把套在导管里的导丝链起点挪到导管头 |
| `beamadapter/build.sh` | BeamAdapter | 问题 1、8、17、22 | 下载上游 `v25.12`（`5d02b83`，和 SOFA 25.12 自带的一致）→ 依次打上五个补丁 → 编译 → 装到 `~/opt/BeamAdapter-v25.12-dev` |

## 怎么生效

- **Python 的修正**：用 stEVE 原类的名字（`SofaBeamAdapter`（`device_contact.py`）、`CenterlineRandom`、`Manual`、`AorticArchRandom`、`Tracking2D`、`NormalizeTracking2DEpisode`、`BruteForceBFS`、`TrackingOnly`（`fluoroscopy.py`））登记到训练框架里，配置文件照常写这些名字，拿到的就是修过的版本。只在上游 stEVE（conda 环境 `guidewire_rl`）下生效；旧环境里不登记，G13 复现不受影响。
- **BeamAdapter**：`guidewire_rl` 环境的 `SOFA_PLUGIN_PATH` 先列 `~/opt/BeamAdapter-v25.12-dev`，SOFA 会先加载这份。SOFA 安装目录里自带的那份没动。改了补丁以后，重跑 `bash simulator_modifications/beamadapter/build.sh`。`beamadapter/build/` 是编译时下载的源码和中间文件，不算正文，随时可以删。
- **直接改 stEVE 源码**：在工作区根目录的 `stEVE/`（`dev` 分支）里改。目前那里一行没改。

## 测试

`tests/test_simulator_modifications.py`，在 `guidewire_rl` 环境里跑：

```bash
conda run --no-capture-output -n guidewire_rl python -m pytest -q tests/test_simulator_modifications.py
conda run --no-capture-output -n guidewire_rl python -m pytest -q tests/test_device_contact.py   # 问题 8、22
```
