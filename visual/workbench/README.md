# workbench：导丝工作台

一个界面里做这几件事：选 phantom 版本和仿真设置、换导丝和导管、改参数，在仿真器里部署策略，看器械不受力时的样子，在没有血管的空中实时推、转器械（器械仿真），看中心线，切到 X 光模式看 C 臂上大概看到什么。浏览器里看，Python 端跑仿真（viser，装在 `guidewire_rl` 环境里）。界面中英文，默认中文，面板最上面的按钮一键切换。

由原来的部署查看器 `deploy_viewer/` 扩出来（2026-10-28），部署那部分照旧。

## 怎么用

```bash
# 默认打开 模型内壁版（头和胸腔）× 摩擦 0.1；phantom、设置、参数、策略都能在界面上换
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py
# 指定 phantom、设置和 checkpoint，一打开就开始跑
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py \
    --phantom spheres_cut --setting phantom_friction --checkpoint runs/<run>/checkpoints/step_N.pt --play
# 器械仿真：没有血管，Glidewire 硬杆穿过 Cordis Sidewinder II，一打开就开始演示
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py \
    --phantom bench --setting two_devices_ica --wire glidewire_stiff --catheter cordis_sidewinder2 \
    --policy 演示：导丝穿过导管 --play
# 回放一回合保存下来的记录（不需要 SOFA）
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py --replay runs/workbench/<文件>.npz --play
```

然后用 Chrome 打开 http://localhost:8080 （要 WebGL；桌面端自带的浏览器窗格没有，画不出来）。

其他参数：`--phantom`（`rebuilt_cut` 模型内壁版、`spheres_cut` 内接球版、`wilson`，带右臂的 `rebuilt`、`spheres`，没有血管的 `bench` 器械仿真）、`--wire` / `--catheter`（换成哪根真实器械，名字见下面“导丝 / 导管”；器械仿真里 `--catheter none` 不要导管）、`--setting`（`phantom_g13`、`phantom_friction`、`phantom_real_mass`），`--env` 打开某个配置对应的 phantom × 设置（`configs/` 里的名字或路径；`--config` 是同一个参数；配置里的血管不在列表里时用默认 phantom），`--set 段.键=值` 临时改配置（之后打开的每个环境都改），`--policy <菜单里的名字>`，`--seed` 第一回合的种子（默认 1000，之后每回合 +1），`--speed` 起始速度（步/秒），`--auto-next` 一回合结束后自动开下一回合（默认停在最后一步），`--device`，`--record-dir`（默认 `runs/workbench/`），`--lang en`，`--port`。

## 界面

- **最上面**：语言切换；状态（第几回合第几步、仿真时间、奖励、尖端离目标多远、送入长度和转角、种子、场景）。两根器械时送入长度和转角每根一行。
- **环境**：两个下拉框，换哪个都重建仿真（几秒到十几秒）。每个下拉框选中的是什么（phantom 怎么来的、设置是哪份配置、器械的数从哪来）不常显示，鼠标浮在下拉框上才出来（2026-11-12 用户要的）。
  - **phantom 版本**（血管 = 碰撞墙 + 中心线；每个版本是什么、怎么来的见 `guidewire_rl/env/vessels/biro_phantom/README.md`）：模型内壁版（头和胸腔，默认）、内接球版（头和胸腔）、Wilson 版（G13 的生成主动脉弓，不是 phantom，对照用）；额外的：模型内壁版和内接球版的完整模型（带右臂）。两个 phantom 版本都只用 STL 生成（2026-10-29 起内接球版也是；lab 输入做的那份内接球版 `lab_cut` / `lab` 还在仿真里，不再列）。都能跑策略。在 `parts/environments.py` 的 `PHANTOMS` 里加一项就多一个；也能放只看形状的（网格 + 中心线文件，`ShapeSource`）。
    - **无血管：器械仿真（空中）**（`bench`）：没有血管，只有导丝和导管在空中，实时推、转（7.5 步/秒就是真实时间，两根器械时每步约 0.07–0.16 s 算得完）。器械按“导丝 / 导管”菜单，物理（物理步、预判距离、质量）用所选设置的，观测、奖励、目标都不用，没有摩擦菜单（没有墙）。两根都从 0 送入、转角 0 开始，这时弯头弯在水平面（x-y）里，相机从正上方往下看，下面衬一张 1 cm 一格的网格（5 cm 一条亮线），橙线是插入方向。策略菜单第一项是**演示：导丝穿过导管**：导管先送出“弯头长度 + 80 mm”、导丝留在弯头后 20 mm，然后导丝穿过去伸出导管头 60 mm、停一下、退回、再穿，一直循环（`parts/bench.py` 的 `ThroughDemo`）；也能用键盘手动推、转。不画尖端走过的路（领先的尖端在两根器械间换，画出来是乱线）。仿真部分是 `guidewire_rl/env/instruments/free_space.py`（`FreeSpace`：训练用的 SofaBeamAdapter、远处一个三角形当“血管”）。
  - **器械设置**（血管和摩擦以外的一切：器械、质量和物理步、观测、奖励、任务终点）：从 `configs/` 读，只差血管或摩擦的几个配置算一个（`phantom_friction`、`phantom_root`、`phantom_all_vessels` 都并进 `phantom_g13`）。现在五个：只有导丝（默认质量、6 ms）、只有导丝（真实质量、2 ms）、导丝 + 钩形导管第一轮（目标右颈总）、导丝 + 钩形导管真实质量（目标右颈总）、导丝 + 钩形导管真实质量（目标右颈内，手写推法 11-06 调的那套；后两个只在 `main` 的代码上有）。两器械的目标按分支和路口定，配 Wilson 版开不了，状态栏显示切换失败；之后选别的照常打开（另一半保留刚才选的）。新加的配置自动出现，名字在 `SETTING_TEXT` 里写。`g13.yaml`、`g13_upstream.yaml` 不是 phantom 的设置，不列。
  - **导丝 / 导管**：默认“按设置”（配置里写的那根），也能换成我们按真实产品做的器械（`guidewire_rl/env/instruments/`，名单在 `catalog.py`，每根的数从哪来：鼠标浮在下拉框上显示）：
    - 导丝：`glidewire_stiff` Terumo Glidewire GS3508 0.035″ × 180 cm 硬杆带角度（lab 用的那根；硬度是实测，和训练配置里手写的数相同）。
    - 导管：`cordis_sidewinder2` Cordis Tempo Aqua Sidewinder II 5F（SRD6109；外径 1.65 mm 按包装，头形是 Qiu 2024 那根 Tempo SIM 2 的照片描线、硬度 486 N·mm² 是那根的实测）；`accuvu_omni_flush` AngioDynamics Accu-Vu Omni Flush 5F（外径 1.8 mm 按包装，弯形、内径、硬度都是估算，等拍实物再换）；`lab_hook` lab 的钩形导管（现在训练用的）。
    - 在血管里只换不增减（任务的观测和动作是按几根器械定的）：只有导丝的设置没有导管菜单；换的器械沿用原来那根的速度上限和名字。器械仿真里可以给只有导丝的设置加一根导管，或选“不要导管”。换器械会重建仿真，参数回到这套设置的值，摩擦保留。
    - 训练配置本身没改（新器械按约定第二轮一起改进配置）。
  - **摩擦**（器械和管壁）：单独选，0.001（G13 原值）或 0.1（lab 的 phantom 灌了水，默认），换器械设置时保留；在参数页改了别的值也会出现在这里。命令行 `--setting <配置名>` 或 `--env` 用那份配置自己的摩擦。
  - **代码从哪来**：环境变量 `WORKBENCH_CODE=<另一个检出>` 时，仿真代码（`guidewire_rl`、`simulator_modifications`）和 `configs/` 从那里读，例如还没提交的改动所在的工作树；`runs/` 和存的回放仍在工作台自己的检出。
  - `configs/` 里的配置用的都是 lab 输入做的血管（lab 原版 6 条线，`phantom_all_vessels` 是 lab 的内接球版完整模型），不在列表里，它们只当"设置"用；`--env` 打开它们时用默认 phantom。任意组合（例如模型内壁版 × 真实质量）都能直接开。
- **运行**：
  - 选策略：随机动作、不动、只往前推、键盘手动、两版手写推法（只在两器械时列出，读仿真真值，`parts/scripted.py`）：**手写推法·11-06（真实质量）**是现在的 `guidewire_rl.env.two_device.script`（11-06 按真实质量重调、能走到右颈内），**手写推法·第一轮（重质量）**是 `guidewire_rl.env.two_device.script_round1`（第一轮训练录示范用的那版，按默认重质量调的）；两版都能配任何两器械设置，例如把第一轮那版放到真实质量的设置上看，以及 `runs/` 下观测和动作与这个环境一致的 checkpoint（看 `runs/<run>/config.yaml`；物理可以不同）。`--checkpoint` 给的总是列出来。
  - **键盘手动**（只在实时仿真时有）：按住移动、松开不动。导丝 ↑/↓ 推 / 拉、←/→ 转；导管 W/S 推 / 拉、A/D 转（只有一根器械时只有方向键）。速度 = "键盘速度"滑条（最大速度的比例，默认 0.5）× 环境的最大速度。先点一下三维画面让页面收到按键，按"播放"后仿真才走。选了它以后这些键不动相机（相机用鼠标）。做法：viser 不把按键交给 Python，侧栏里有一小段脚本在浏览器里听键，每 0.2 s 把按着的键报给工作台自己开的一个小 HTTP 端口（`parts/keyboard.py`）；0.6 s 没收到就当松开，切走标签页器械就停。
  - 播放 / 暂停、单步、重新开始；速度（7.5 步/秒等于真实时间）；结束后自动开下一回合（默认不勾：一回合结束就停在最后一步，成功时能看清最后的样子，按“重新开始”开下一回合；2026-11-13 用户要的，命令行 `--auto-next` 一打开就勾上）；下一回合的种子；回看第几步。
- **目标点**：stEVE 把目标存在透视坐标系里（转过 `image_rot_zx` 的），2026-10-28 前工作台直接拿来画，目标球和"尖端到目标"距离偏了约 30 mm；现在先转回血管坐标系（`sources.target_in_vessel_frame`）。训练本身不受影响（环境自己算奖励时坐标一致）。
- **器械**标签页：每根器械不受力时的样子（末端按比例画，线宽 = 粗细；下面一条是全长，灰 = 杆身、红 = 软段），以及全长、质量、弯头角度和半径、每段的长度、外径、杨氏模量、密度。读的是仿真里实际建的器械，所以换了导丝或改了质量都会跟着变。
- **参数**标签页：摩擦系数、物理步、碰撞预判距离、导丝杆身和软段密度、C 臂角度、到达目标的距离、每回合步数上限。显示的是这套设置里的值（配置没写的用类的默认值），改了按"应用（重建仿真）"生效，改过的那项标"已改"，状态栏的场景一行写 ✎ 和改了哪几项。只在工作台里生效，不动配置文件；换 phantom 时保留，换设置或按"恢复这套设置的值"时回到设置里的值。改物理步时受力标签的小步数跟着改。只看形状的 phantom 只能改 C 臂角度。在 `parts/parameters.py` 的 `PARAMS` 里加一项就多一行。
- **曲线**标签页（第一页）：顶上是这一步的输出（每根器械的推送、旋转：归一化值和 mm/s、rad/s）和交互力（导丝尖端、导管尖端的接触力，实测 / 模型预测；离壁余量那些不是力的标签不显示）；下面是曲线：导丝动作、导管动作、导丝尖端接触力、导管尖端接触力（导管的两张只在有导管时显示）、奖励、尖端到目标的距离。观测（策略的输入）和控制器看到的透视小图不再显示（2026-11-06 用户：太多了没必要）。
- **显示**标签页：
  - **X 光**：
    - X 光模式：沿环境里 C 臂的方向看（`image_rot_zx`），视角很窄、离得很远，近似平行投影，和策略的 2D 画面一致；背景换成灰色，导丝画成黑的。默认看不到血管（没打造影剂），中心线、管子、尖端轨迹都不画。
    - 打造影剂（路图）：血管画成半透明的灰影。两根血管在画面上前后叠住的地方会有斑点，是浏览器画半透明面的限制，不是网格的问题（两面墙都是封闭、朝向一致的）。
    - 标出策略拿到的东西：策略跟踪的 5 个点（按画面上 1 mm 间隔从尖端取）和目标。
    - 点"视角"里任何一个按钮都会退出 X 光模式。
  - **图层**：血管、人体轮廓、训练用的 6 条中心线和插入点、其余血管的中心线、中心线半径（每条中心线按记的半径画成管子，看观测和奖励以为的管腔和碰撞墙差多少）、器械、尖端走过的路、目标、策略跟踪的点、受力箭头；血管不透明度、人体不透明度；导丝按受力变色。
  - **路口地图**：6 个绿色小球，去右颈内那条路线上有血管分出去的地方（第二轮设计第 1 块定的，`WORK_LOG` 2026-11-08 01:33）：左锁骨下口、左颈总口、头臂干口、头臂干末端、颈总 64 mm 侧支口、颈内 / 颈外分叉。位置按路线代码 `two_device.route.VesselGraph` 算：从主动脉分出去的三支取它离开主动脉管腔的地方，其余取分支起点（`parts/junctions.py`）。只在 BIRO phantom 上画；X 光模式下勾了"标出策略拿到的东西"才显示（它是观测的一部分）。方向没画。
  - **人体轮廓**：半透明的躯干和头（MakeHuman 的基础人体网格，CC0，`assets/`；只有皮肤表面，没有骨骼；手臂截在肩膀外 6 cm、腿截在胯部，截口封平），看血管大概在身体哪里。人形保持原样，不拉伸、不扭脖子，只整体挪动和转动：左右按脖子里两根颈总动脉、高度按主动脉弓顶在胸骨上缘下 25 mm、前后按胸腔里最靠前的血管在皮下 20 mm 先摆正，再绕弓顶整个往前倾 20°、往下 40 mm、往前 60 mm（phantom 的颅内血管比站直的人靠前）。血管点全在人形里，九成离皮至少 17.6 mm。只在 BIRO phantom 上有（场景的主动脉和 phantom 的对得上才画；Wilson 版没有），X 光模式里不画。只画朝里的那面（从外面看是远处的后壁），所以挡不住血管和导丝。
  - **器械**：只有导丝时照旧画一根 1.8 mm 的管子。导丝 + 导管时，仿真里两根套在一起、存下来只有一串点，按各自的送入长度拆开（短的那根从领先的尖端往回数差值，同两器械环境的算法）：导管画成蓝色、按外径（2 mm），导丝按外径（至少 1 mm）、按受力变色，只画伸出导管的那段（缩在导管里时看不到，尖端球也藏起来）。
  - **视角**：透视机方向 / 侧面 / 尖端附近，相机跟着尖端走。
  - **保存这一回合**。

## 结构：哪些是能换的零件

| 文件 | 管什么 | 接口 | 现在有的 |
| --- | --- | --- | --- |
| `parts/environments.py` | 能打开什么：phantom 版本 × 仿真设置 | `phantoms()`、`settings()`、`build_config(phantom, setting)`、`locate(配置)`；`compatible(环境配置, run 配置)` 判断 checkpoint 能不能用 | `PHANTOMS` 6 个（3 个主版本 + 2 个带右臂的：`rebuilt_cut`、`spheres_cut`、`wilson`、`rebuilt`、`spheres`；没有血管的 `bench`）；设置从 `configs/` 读；`with_devices(配置, phantom, 导丝, 导管)` 换器械 |
| `parts/parameters.py` | 参数面板 | `PARAMS`（每项指向配置里一处）；`read`、`apply`、`changed` | 8 项 |
| `parts/frame.py` | 一步的数据 `Frame`、一回合不变的 `SceneInfo`（含训练分支名单、中心线半径、器械）、能存能读的 `EpisodeLog` | — | 存成一个 `.npz`（`workbench.episode/2`，也能读旧的 `deploy_viewer.episode/1`） |
| `parts/bench.py` | 器械仿真：没有血管，器械在空中 | 同 `FrameSource`；`bench_menu(source)` 策略菜单；`ThroughDemo` 穿过演示 | `BenchSource`（观测 = 每根的送入长度和转角，动作 = 每根的推、转速度占上限的比例） |
| `parts/sources.py` | 数据从哪来 | `FrameSource`：`scene_info()` `reset(seed)` `step()` `close()`，`can_run` | `LiveEnvSource`（按 YAML 起仿真器）、`ReplaySource`（读存下来的回合）、`ShapeSource`（只看形状） |
| `parts/policies.py` | 动作从哪来 | `Policy`：`reset()` `act(obs, first_step)` `predict(obs, action)` `debug()` | 随机动作、不动、只往前推、`CheckpointPolicy` |
| `parts/keyboard.py` | 键盘手动 | `KeyboardPolicy`（同上）；`KeyServer` 收浏览器报的键，`capture_html` 是侧栏里那段听键的脚本 | 导丝方向键、导管 W A S D |
| 同上 | 动作出来以后再改一道 | `SafetyFilter.filter(obs, action, predictions)`，用 `Filtered(policy, filter)` 套上 | `PassThroughFilter`（不改） |
| `parts/instrument.py` | 器械窗 | `instrument_info(stEVE 器械)` → `InstrumentInfo`；`instruments_html` | 读 stEVE 的 `base_section` + `tip_section`（直段、网格段、螺旋段） |
| `parts/body.py` | 人体轮廓 | `BodyLayer`（同其他图层）；`body_mesh()` 摆好的网格，`trunk()` 截掉手脚，`fits(info)` 这个场景画不画 | MakeHuman 基础网格（`assets/makehuman_base.obj`，来源和授权见 `assets/README.md`）的躯干和头，按 `rebuilt_cut.json` 的血管摆 |
| `parts/scene.py` | 三维图层 | 每层 `set_scene(info)` `update(...)` `set_visible()` | 血管、两组中心线、半径管子、器械（导丝；有导管时分开画）、轨迹、目标、跟踪点、受力箭头 |
| `parts/xray.py` | X 光模式 | `XrayMode.enter/leave/apply_camera`；`visible_in_xray` 定每层在 X 光模式下显不显示 | 灰背景、窄视角相机、造影外观 |
| `parts/text.py` | 侧栏文字 | 状态表、`action_html`（输出，两根器械时每根两行）、`force_html`（交互力）；名字表 `ACTION_TEXT`、`FORCE_TEXT`，每项写 (中文, English)；没写的受力显示原名 | 导丝、导管的尖端接触力 |
| `parts/i18n.py` | 界面上其余的字 | `TEXT` 表，每项 (中文, English)；`tr(键, 语言)` 取字 | 中文、英文 |
| `parts/plots.py` | 曲线 | `PLOTS` 里加一项就多一张图；`SHOWN_IF` 定哪张只在有某样东西时显示 | 6 张（导丝、导管各两张，奖励、距离） |
| `workbench.py` | 把上面这些拼起来，加播放控制和换环境 | — | — |

`Frame.extras` 是给以后的模块留的空位，已经会读：`nominal_action`（安全滤波改动之前的动作）、`filter`、`force_vector_N`（有了就画受力箭头）、`error`。

## 还没接上的（接口已经留好）

1. **受力方向**：`distal_force` 标签只给大小，受力箭头暂时不显示。
2. **G16.1 模糊安全滤波**：写一个满足 `SafetyFilter` 接口的类，用 `Filtered` 套到策略上。
3. **规划器内部**（MPPI 候选动作、精英序列）：由 `Policy.debug()` 交出来，放进 `extras`。
4. **真机或实体 phantom**：再写一个 `FrameSource`。
5. 旧环境 `contact_aware_tdmpc2` 没装 viser，G13 原版那套仿真器看不了。
6. X 光模式是按几何画的样子，不是按衰减算出来的 X 光图像。

## 测试

```bash
conda run --no-capture-output -n guidewire_rl python -m pytest -q visual/workbench/tests             # 约 7 s，含三个真起 SOFA 的
conda run --no-capture-output -n guidewire_rl python -m pytest -q visual/workbench/tests -m "not sofa"
```
