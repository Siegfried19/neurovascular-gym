# 实验室 phantom 的血管数据

## phantom 版本一览（2026-10-28 定）

实物 phantom 只有头和胸腔；模型（STL）多一条右臂。工作用的三个版本是 **模型内壁版**、**内接球版** 和 **Wilson 版**（前两个用只有头和胸腔的截断版）；带右臂的完整模型是额外的。代码和文件名里模型内壁版叫 `rebuilt`，内接球版叫 `spheres`，两个都只用 STL 生成（内接球版 2026-10-29 起）；lab 输入做的那份内接球版叫 `lab`，留着对照，工作台不再列。仿真里用 `BiroPhantomArch(version=...)` 选（配置写 `vessel_tree: {type: BiroPhantomArch, version: rebuilt_cut}`），工作台的"phantom 版本"下拉框也是这几个。

| 版本 | `version` | 碰撞墙 | 中心线 | 路口 | 文件 |
| --- | --- | --- | --- | --- | --- |
| 模型内壁版（头和胸腔） | `rebuilt_cut` | 模型自己的内壁，右臂截掉 | 在这面墙里按"尽量离墙远"重算，60 段 | 文件里写明（按线怎么连） | `rebuilt_cut.obj`、`rebuilt_cut.json` |
| 内接球版（头和胸腔） | `spheres_cut` | 沿模型内壁版的中心线放内接球（半径至少 1 mm）拼起来，右臂截掉 | 同模型内壁版，半径抬到至少 1 mm，60 段 | 文件里写明 | `spheres_cut.obj`、`spheres_cut.json` |
| Wilson 版 | —（`AorticArch`, seed 30） | stEVE 生成的主动脉弓，不是 phantom | stEVE 自带 | stEVE | 无文件，stEVE 现生成 |
| 模型内壁版（完整，带右臂） | `rebuilt` | 模型自己的内壁 | 同上，61 段 | 文件里写明 | `rebuilt_full.obj`、`rebuilt_full.json` |
| 内接球版（完整，带右臂） | `spheres` | 同上，不截 | 同上，61 段 | 文件里写明 | `spheres_full.obj`、`spheres_full.json` |
| 对照：lab 输入的内接球版（头和胸腔） | `lab_cut` | lab 的墙（沿 lab 中心线放的内接球拼起来），右臂截掉 | lab 的线补全（每条血管一条，主动脉到根部底），74 段 | stEVE 按半径猜（lab 的做法） | `lab_cut.obj`、`lab_cut.json` |
| 对照：lab 输入的内接球版（完整） | `lab`（= `centerline="all_vessels"`） | lab 的内接球墙 | 补全的 75 段 | stEVE 按半径猜 | `phantom_arch.obj`、`phantom_arch_all.json` |

**模型内壁版**只读一个输入：phantom 模型 `biro-phantom_vessel-wall_v2.stl`（sha256 `d75daffc…67f3`，本机在 `reference/contact-aware-tdmpc2-training/VasselTree/Neurovascular model.stl`），其余全是我们的代码（`../biro_rebuild.py`），不用 lab 的任何派生文件。手定的只有三样：训练坐标系（lab 47 拟合的绕 z 转 37.67°、原点）、lab 47 的任务定义（主动脉从第一个分支前 146.7 mm 开始；侧支截到论文测试弓的长度 rcca 37.8、rsa 42.4、lcca 58.9、lsa 58.9 mm；头臂干到右颈总分出为止）、和脚本开头列的算法参数。步骤：
1. 内壁：一个面算内表面，当沿它法线 0.4–1.4 mm 处的点在三个坐标轴里至少两个方向上两边都被壁挡住（lab 18 判体素的同一条规则，这里用沿轴的射线判断）；取最大的一块，36 个开口用扇形封上，减到 12 万个三角形，挪进训练坐标系。和以前借 lab 体素做的新墙比，多收进了远端冠状动脉（lab 的体素没填满那里）。
2. 中心线：内腔填 0.3 mm 格点，代价 1/R²（R = 到墙的距离），从入口（最宽的开口）一次 Dijkstra 到每个开口；没被线盖住的内腔里，连着两处的是环（Willis 环一带，接上 4 条），连着一处、够长的是封死的血管（补线，11 条），其余是鼓包不管；主动脉接到根部底（根部那块腔里 R ≥ 1 mm 的最低点）。每段平滑、重采样，半径 = 到墙的精确距离（开口端不算封口面，至少 0.3 mm）。
3. 血管和名字：路口处粗细明显（1.25 倍）占优的接着走，差不多时排除 30 mm 内就封死的短支、取最直的；主动脉是从入口出来的那条，弓顶 40 mm 内最粗的三条分支从入口算起依次是左锁骨下、左颈总、头臂干；头臂干分叉处往上走得多的是右颈总。
4. 截断（`../biro_cut.py`，两种墙同一规则）：lab 没截过手臂（它的内腔、球拼墙、血管树都带着），实物右侧胸腔到哪也没记录，所以按左右对称：左锁骨下从主动脉开口到它的开口端有多长（lab 输入的版本 91.8 mm，我们的中心线 98.6 mm），右锁骨下就从头臂干分叉处量同样长，在那里用垂直于中心线的平面截断、封平；之后的线（前臂那条）删掉，之前分出的右椎动脉等都留着；冠状动脉在胸腔里，留着。

**内接球版**（`../biro_spheres.py`，约 7 秒）只读模型内壁版的中心线 `rebuilt_full.json`，墙的拼法照 lab 19 的 `mesh.py`：沿每条中心线每 0.5 mm（中位半径 ≥ 3 mm 的每 1 mm）放一个内接球，半径至少 1 mm（导丝的接触模型离墙 0.3 mm，要留空间；lab 也这样）；0.5 mm 格点上取"到最近球面还有多深"，等值面画成三角网格，减到 6 万个三角形（保体积），只留最大一块（减面会在表面留一个两三角形的扁片，lab 的 `phantom_arch.obj` 在 [11.5, 1.0, 23.0] 也有一个）。中心线、路口、插入点照搬模型内壁版，半径抬到至少 1 mm。和 lab 输入的内接球版比（尺子是模型内壁围出的真实管腔 256.6 ml）：墙 235.4 / 208.4 ml，铺满真实管腔 92 % / 81 %（训练用 6 条：95 % / 86 %）；训练用 6 条上墙比真实的壁往里缩中位 0.24 / 0.71 mm，两边都不出墙；lab 的半径量到它的方块管腔（0.5 mm 体素，比真实的小一圈），所以整体瘦一圈。

我们的中心线和 lab 的比（训练用的 6 条）：线走的是同一条路，每点离 lab 线中位 0.16–0.39 mm；差别在分支起点，挪了 6–9 mm（我们的起点是"通往两个尽头的中心线分开处"，lab 是骨架交点，都在主干管腔里）；所以插入点沿降主动脉挪了约 7.9 mm（主动脉起点按左锁骨下开口往前 146.7 mm 定）；主动脉根部底的位置差 15.6 mm（真实内壁的根部最低处和球拼墙不在一处）；半径大 0.5–1.1 mm（lab 的半径量到比真实小一圈的方块管腔）。

重新生成（结果逐字节相同，测试用下表的哈希锁着）：

```bash
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_rebuild   # 模型内壁版两份，约 2 分钟，要 STL
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_spheres   # 内接球版两份，约 7 秒，读上一步的中心线
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_cut       # lab 输入的内接球版截断，几秒
```

| 文件 | sha256 |
| --- | --- |
| `rebuilt_full.obj` | `e3d22f5b375749a15fca0366968d8f0cfafd369bac1c5d566bae5ef37e4f300c` |
| `rebuilt_full.json` | `e36bcf3f37605ea29c3d4f7ac2f6f0b172027d203444c24c576486039efede33` |
| `rebuilt_cut.obj` | `a358f8f5714679f6f1062119cf262e8d33b5a9cec241da8c6c17039f09c399a1` |
| `rebuilt_cut.json` | `177a796db47ca1a3827629f1bd6c86cc08480fffbc664559d0240cdf35cb53a9` |
| `lab_cut.obj` | `bca7b783970b180307052740f1d9a3257e513f54d8cb962675c00d94575468b5` |
| `lab_cut.json` | `945de2eca4c078494bff7d2943c06da933abbf67efdd15e5ad7be3e6a6c280ce` |
| `spheres_full.obj` | `e336d97d24b9a29bf2c9735335b4981f109a8e59c461f0ba1f030f63753bcafe` |
| `spheres_full.json` | `2116e87bf6b289151ec68984727cef4a3ed4e1fb107e065b034d4a73efef9b58` |
| `spheres_cut.obj` | `97aea5548cd0fa9853c0120b40f208b965db512618dbdcd4622f2683121caa64` |
| `spheres_cut.json` | `f5e96faba09d7674f0a068d4e2e6e0287f83f1dfc3bad78d172c0a3dd4f144e0` |

被取代、等用户定删不删的：`phantom_lumen.obj` / `phantom_lumen.json`（借 lab 体素做的新墙，模型内壁版的墙取代它）、`phantom_lumen_all.json`（配它的中心线，量到新墙的半径）。lab 原版 6 条线（`phantom_arch.json`）和根部延长（`phantom_arch_root.json`）不再单独作为版本，但它们是 lab 输入的内接球版的原料，`configs/` 里的配置也还在用，留着。

## 下面是各份文件的来历（2026-10-24 起）

训练用的 phantom 血管（`BiroPhantomArch`，见上一级的 `biro_phantom.py`）读下表的文件。前两份是 2026-10-24 从 lab 仓库 `ALL-IN-EXO/surgery-all-in-one` 原样复制来的：仓库 HEAD 是 `9c70d88`，这两份文件最后一次修改在 `41b32b2`（2026-10-16）。原位置是 `47-steve-agent-phantom/derived/`。

| 文件 | 内容 | 生成它的 lab 脚本 | sha256 |
| --- | --- | --- | --- |
| `phantom_arch.json` | 6 条分支（aorta、bct、rcca、rsa、lcca、lsa）的中心线和每个点的半径、插入点和插入方向，以及摆放变换（绕 z 转 37.67°、平移量） | `47-steve-agent-phantom/phantom_arch.py` | `70595a2e046f914ba397624751d24b54a8231673dc3f8b5d0cab346429dcea8a` |
| `phantom_arch.obj` | SOFA 碰撞网格：整个 phantom（69 条血管，含头部、手臂），已用同一个变换挪好 | `19-phantom-sofa/mesh.py` 生成，再经 `47-steve-agent-phantom/phantom_arch.py` 挪动 | `e23430efa986bd93734e7ec613d97a676c35f9e163fb9b86f6731c2906a5db4a` |
| `phantom_arch_root.json` | 我们做的：同 `phantom_arch.json`，只是主动脉中心线延到了根部的底（见下文）。`BiroPhantomArch(centerline="root_extended")` 读它，`configs/phantom_root.yaml` 用它 | 我们的 `../biro_phantom_root.py`（由上面两份算出，2026-10-27） | `2a2507f0f0088cc1110340f20d90038ed0a5a5f44fb44b8d2fd91db894753063` |
| `vessel_tree.npz` | lab 的全部 69 条血管（中心线点和半径，phantom STL 坐标），碰撞网格就是沿这些线放球拼的。从 lab 仓库原样复制（2026-10-27，HEAD `8f102e1`，文件最后一次修改在 `41b32b2`），原位置 `19-phantom-sofa/derived/tree.npz` | `19-phantom-sofa/tree.py` | `0c5bebc39cfdcf37057e97301e53feafb1e33a38ccf87e1851ff4b336c01abc6` |
| `phantom_arch_all.json` | 我们做的：`phantom_arch_root.json` 的 6 条加上其余每条血管的中心线（共 75 条，见下文）。`BiroPhantomArch(centerline="all_vessels")` 读它，`configs/phantom_all_vessels.yaml` 用它 | 我们的 `../biro_phantom_all.py`（由上面几份算出，2026-10-27） | `acdd043a931a1365a3236ba0be44ed004d3774d34b23cd926755c856c5c447e8` |
| `phantom_lumen.obj`, `phantom_lumen.json` | 我们做的：phantom 模型自己的内表面做成的碰撞面（12 万个三角形，封闭，256 ml；lab 的球拼墙 208 ml），和它的记录（来源哈希、开口、偏差）。还没接进训练 | 我们的 `../biro_phantom_lumen.py`（由 phantom 模型 STL 和 lab 18 的体素内腔算出，2026-10-27；两份源文件只在本机 `reference/`） | 见 `phantom_lumen.json` |
| `phantom_lumen_all.json` | 我们做的：配 `phantom_lumen.obj` 用的中心线。线和 `phantom_arch_all.json` 一样（少 v45、v46 两段 1 mm 长的碎段），主动脉根部按新墙重算，每个点的半径改成到新墙的距离（至少 0.3 mm）。还没接进训练 | 我们的 `../biro_phantom_lumen_centerlines.py`（2026-10-27） | `4c642ffa38a337e90204332a1da73a3c316bcfe5b9fcc36097bd1690459f3ca7` |

坐标系：z 指向头，主动脉起点在原点，单位 mm。中心线从入口往远端排。

lab 那边的说明（`47-steve-agent-phantom/README.md`、`19-phantom-sofa/README.md`）提到的已知问题：

- 侧支被截短到论文血管的长度，每条 40–60 mm；
- 碰撞网格由中心线半径拼出来，只填满真实内腔的约 88 %，细血管放宽到 1 mm；
- 碰撞网格里有 6 条分支以外的血管，导丝进去以后，几何量是对着最近那条已知血管算的。
  - 2026-10-27 处理（`phantom_arch_all.json`）：6 条中心线拼的管子只盖住碰撞网格内腔的约 72 %。其余 69 条（lab 的 64 条血管整条，加上主动脉入口那 70.7 mm、4 条目标分支截断处以后的部分）按 lab 的变换挪进训练坐标系、半径取 max(lab 半径, 1 mm)（和碰撞网格一样），作为额外分支加进去，管子盖住约 98.5 %。这 6 条仍是任务分支（`task_branches`）：目标点和 2D 坐标范围只按它们算，和原来逐位相同；离壁距离、奖励里的路程、stEVE 停推的位置看全部分支。以前 4 条目标分支（rsa、rcca、lcca、lsa）的截断处被 stEVE 当成血管尽头，往前推被清零；现在不再是。

我们自己查到的（2026-10-24）：

- **主动脉根部那头，中心线比血管腔短约 16 mm。** 主动脉中心线最后一个点在 z = 20.6，半径从 13 mm 收到 9.8 mm；碰撞网格却一直到 z = 4.4。封闭的管子提取中心线时，中心线本来就会在离端头约一个半径的地方停下。导丝推过头进了这一段时，离壁距离会算成负的（约 −6 mm）。随机动作走 1077 步里，有 138 步是这种情况；另有 7 步出现在主动脉弓上，约 −1 到 −2 mm，是那里的碰撞网格比中心线半径宽。
  - 2026-10-27 补充：lab 18 的骨架在 z ≈ 20.5 分成两条冠状动脉，主动脉本身没有往下的一段；碰撞网格的根部大腔底在 z 4.4（lab 18 的真实内腔底在 z 4.0），是其中一条冠状动脉开头那段的球撑出来的。stEVE 把中心线末端当作血管尽头：导丝头越过末端点后，往前推的指令被清零（`MonoPlaneStatic` 的 `stop_instruments_at_tree_end`，默认打开）。但它挡不住导丝：导丝头会贴着侧壁滑到根部的底（这时最近的中心线点还不是末端点），越过末端点以后往前推才一直被清零，只能退。
  - 处理（`phantom_arch_root.json`，`configs/phantom_root.yaml`）：按中心线的标准定义（穿过内腔、尽量离墙远的路，每点半径 = 到墙的距离；墙就是上面的碰撞网格）重算主动脉末端。lab 的线保留前 168 个点（和重算的线相差不到 0.3 mm），最后 4 个点（偏 0.4–2.1 mm，朝冠状动脉分叉口去而不是朝底）换掉，接上 15 个新点，止于根部底上方 1 mm（z 5.5，半径 1.0 mm）。原文件和碰撞网格不动；分叉点、目标候选点、2D 坐标范围都不变。同一串偏向往前推的随机动作（10 局、各 200 步，摩擦 0.1）：原中心线下导丝头在根部 1129 步，其中 1105 步往前推被清零，543 步离壁距离为负，最负 −7.2 mm；延长后在根部 545 步，53 步被清零（已到底），78 步为负，最负 −1.2 mm。剩下的负值都在根部侧面另一条冠状动脉的入口那块，那里没有中心线。

这些是 lab 的数据，2026-11-10 经 lab 同意随本仓库公开（来源和授权见根目录 `THIRD_PARTY_NOTICES.md`）。文件不要改；要换版本的话，重新复制并更新上表和 `tests/test_biro_phantom.py` 里的哈希。`phantom_arch_root.json` 随之用 `conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_phantom_root` 重新生成（约 1 分钟，结果逐字节可重复），再更新它的哈希；`phantom_arch_all.json` 接着用 `python -m guidewire_rl.env.vessels.biro_phantom_all` 重新生成（几秒，同样可重复）。
