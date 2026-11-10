# 第三方代码和数据的来源与授权

本仓库自己的代码按根目录 `LICENSE`（MIT）。下面是从别处搬来、或由别处的东西生成的部分，逐项列出在哪、来自哪、什么授权。

## 代码

| 在本仓库的位置 | 来自 | 授权 |
| --- | --- | --- |
| `guidewire_rl/model/agent.py`、`guidewire_rl/model/world_model.py` | Wilson Wan 的 contact-aware TD-MPC2（`wilsonwan8558-netizen/contact-aware-tdmpc2`，`contact_aware_tdmpc2/tdmpc2/`），其底子是 Nicklas Hansen 等的 TD-MPC2（`nicklashansen/tdmpc2`，MIT） | TD-MPC2 原版 MIT；Wilson 的改动经作者同意随本仓库公开（2026-11-10） |
| `guidewire_rl/env/labels/force_reconstruction.py`（原样）、`guidewire_rl/env/labels/distal_force.py`（改写）、`guidewire_rl/env/observations/vessel_geometry.py`（算法照搬） | 同上仓库 `contact_aware_tdmpc2/env/` | 经作者同意（2026-11-10） |
| `guidewire_rl/env/instruments/hook.py`（导管尺寸）、`guidewire_rl/env/two_device/script.py`（手写推法的流程和几个几何判断）、`simulator_modifications/hollow_catheter.py`、`simulator_modifications/speedups.py` | BIRO 实验室仓库 `ALL-IN-EXO/surgery-all-in-one`（`19-phantom-sofa/arch_catheter.py`、`51-legobot-ros2/legobot_sofa/`），按文件开头的说明改写 | 经实验室同意随本仓库公开（2026-11-10） |
| `simulator_modifications/beamadapter/official_irc_fix.patch`、`official_rod_straight_fix.patch` | SOFA BeamAdapter 官方仓库（`sofa-framework/BeamAdapter`）的两处修复，摘成补丁 | LGPL-2.1（BeamAdapter 的授权；补丁只随本仓库分发，编译时打在上游源码上） |
| `simulator_modifications/beamadapter/` 其余四个补丁 | 我们自己写的，改 BeamAdapter | 补丁本身 MIT；打上后的 BeamAdapter 整体仍是 LGPL-2.1 |

不在本仓库里、但运行必需的：SOFA v25.12（LGPL-2.1，`sofa-framework/sofa`）；stEVE（MIT，Lennart Karstensen，`lkarstensen/stEVE`，用上游提交 `c3704cc`，一行没改）。

## 数据

| 在本仓库的位置 | 是什么 | 来自 | 授权 |
| --- | --- | --- | --- |
| `guidewire_rl/env/vessels/biro_phantom/rebuilt_*`、`spheres_*` 和 `guidewire_rl/env/c_arm/data/angles_rebuilt_cut.json` | 由 BIRO 实验室的神经血管 phantom 模型（STL，不在仓库里）用本仓库代码生成的碰撞墙、中心线、C 臂角度表 | phantom 模型属于 BIRO 实验室 | 经实验室同意公开（2026-11-10） |
| `guidewire_rl/env/vessels/biro_phantom/phantom_arch.json`、`phantom_arch.obj`、`vessel_tree.npz`（原样），`lab_cut.*`、`phantom_arch_root.json`、`phantom_arch_all.json`、`phantom_lumen*`（由它们生成）和 `c_arm/data/angles_all_vessels.json` | 实验室自己做的 phantom 中心线、球拼碰撞墙、血管树，以及我们由此派生的版本 | `ALL-IN-EXO/surgery-all-in-one`（`47-steve-agent-phantom/derived/`、`19-phantom-sofa/derived/`） | 经实验室同意公开（2026-11-10） |
| `guidewire_rl/env/instruments/data/simmons.csv` | 一根真实 Simmons 2 导管照片描出的中心线 | Qiu M. 2024, *Energy barriers govern catheter herniation during endovascular procedures: a 2.5D vascular flow model analysis*, Zenodo 记录 11203294（doi:10.5281/zenodo.11203294） | CC BY 4.0；详见同目录 `README.md` |
| `visual/workbench/assets/makehuman_base.obj` | MakeHuman 的基础人体网格 | `makehumancommunity/makehuman`，`makehuman/data/3dobjs/base.obj` | CC0 1.0；详见同目录 `README.md` |

每份数据文件的哈希、生成命令和已知问题在它所在目录的 `README.md`。
