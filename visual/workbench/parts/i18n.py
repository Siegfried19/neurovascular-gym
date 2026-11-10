"""Interface text in Chinese and English; the workbench switches between them at run time.

Every entry is a ``(中文, English)`` pair. ``tr(key, lang)`` picks one and fills
``{placeholders}``. Text that is not in a table (a new observation term, a
checkpoint name) is shown as it is, in both languages.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

LANGS = ("zh", "en")
Pair = Tuple[str, str]

TEXT: Dict[str, Pair] = {
    # panel and run controls
    "title": ("导丝工作台", "Guidewire Workbench"),
    "switch_to": ("English", "中文"),          # the toggle shows the language it switches to
    "starting": ("正在启动仿真…", "Starting the simulator…"),
    "open_url": ("打开 {url}  （Ctrl+C 退出）", "Open {url}  (Ctrl+C to quit)"),
    "environment": ("环境", "Environment"),
    "phantom_pick": ("phantom 版本", "Phantom version"),
    "env_hint": ("换一个会重建仿真（几秒到十几秒）", "Switching rebuilds the simulator (a few seconds)"),
    "setting_pick": ("器械设置", "Instrument setting"),
    "friction_pick": ("摩擦（器械和管壁）", "Friction (instruments on the wall)"),
    "friction_hint": ("换摩擦只重建仿真，其他改过的参数保留", "Rebuilds the simulation; other edited parameters stay"),
    "setting_hint": ("器械、质量和物理步，以及观测、奖励、任务终点；摩擦在下面单独选。换设置时参数回到这套设置的值（摩擦保留）",
                     "Instruments, mass and physics step, and observation, reward, task end; the friction is chosen "
                     "below. A new setting resets the parameters to its own values (the friction stays)"),
    "setting_also": ("用这套设置的还有：{names}（只是血管不同）", "Same setting in: {names} (only the vessel differs)"),
    "parameters": ("参数", "Parameters"),
    "edited": ("（已改）", " (edited)"),
    "no_params": ("回放时没有参数可改", "Nothing to edit in a replay"),
    "apply_params": ("应用（重建仿真）", "Apply (rebuild the simulator)"),
    "reset_params": ("恢复这套设置的值", "Back to the setting's values"),
    "params_hint": ("改了只在工作台里生效，不改配置文件；换 phantom 时保留，换设置时恢复。",
                    "Edits live in the workbench only, the config files stay; kept when the phantom changes, "
                    "reset when the setting changes."),
    "shape_only": ("只看形状：没有仿真，不能跑策略", "View only: no simulator, no policy can run"),
    "run": ("运行", "Run"),
    "policy": ("策略", "Policy"),
    "keys_speed": ("键盘速度（最大速度的比例）", "Keyboard speed (share of the top speed)"),
    "keys_legend_two": ("按住移动：导丝 ↑↓ 推 / 拉、←→ 转；导管 W S 推 / 拉、A D 转。先点一下三维画面；"
                        "这些键这时不动相机（相机用鼠标）。按播放后才走。",
                        "Hold to move: guidewire ↑↓ push / pull, ←→ rotate; catheter W S push / pull, A D rotate. "
                        "Click the 3D view first; these keys do not move the camera now (use the mouse). "
                        "Runs once Play is on."),
    "keys_legend_one": ("按住移动：导丝 ↑↓ 推 / 拉、←→ 转。先点一下三维画面；方向键这时不动相机。按播放后才走。",
                        "Hold to move: guidewire ↑↓ push / pull, ←→ rotate. Click the 3D view first; the arrow "
                        "keys do not move the camera now. Runs once Play is on."),
    "keys_held": ("按着：", "held:"),
    "play": ("播放", "Play"),
    "pause": ("暂停", "Pause"),
    "step": ("单步", "Step"),
    "restart": ("重新开始", "Restart"),
    "speed": ("速度 步/秒", "Speed (steps/s)"),
    "speed_hint": ("7.5 = 和真实时间一样快（仿真慢时达不到）",
                   "7.5 = real time (not reached when the simulation is slow)"),
    "auto_next": ("结束后自动开下一回合", "Start the next episode automatically"),
    "next_seed": ("下一回合的种子", "Seed of the next episode"),
    "next_seed_hint": ("决定目标点和导丝初始转角；每开一回合自动 +1",
                       "Sets the target and the wire's initial rotation; +1 for every new episode"),
    "scrub": ("回看第几步", "Review step"),
    "scrub_hint": ("拖动会暂停并显示那一步", "Dragging pauses and shows that step"),
    "latest": ("回到最新", "Back to latest"),
    # tabs
    "tab_instrument": ("器械", "Instrument"),
    "tab_curves": ("曲线", "Curves"),
    "tab_display": ("显示", "Display"),
    "action_heading": ("**动作**（策略的输出）", "**Action** (policy output)"),
    "force_heading": ("**交互力**（仿真里量的，策略看不到）",
                      "**Contact force** (measured in the simulation, not seen by the policy)"),
    "layers": ("图层", "Layers"),
    "layer_vessel": ("血管", "Vessel"),
    "layer_body": ("人体轮廓（只在 BIRO phantom 上有）", "Body outline (BIRO phantom only)"),
    "layer_centerlines": ("训练用的中心线和插入点", "Training's centerlines and insertion point"),
    "layer_other_centerlines": ("其余血管的中心线", "Other vessels' centerlines"),
    "layer_radius_tubes": ("中心线半径（画成管子）", "Centerline radii (as tubes)"),
    "layer_tracked": ("策略拿到的跟踪点", "Points the policy tracks"),
    "layer_wire": ("器械（导丝；有导管时导管画成蓝色粗管）", "Instruments (guidewire; a catheter as a thick blue tube)"),
    "layer_trail": ("尖端走过的路", "Tip trail"),
    "layer_target": ("目标", "Target"),
    "layer_force": ("受力箭头（还没有数据）", "Force arrow (no data yet)"),
    "opacity": ("血管不透明度", "Vessel opacity"),
    "body_opacity": ("人体不透明度", "Body opacity"),
    "color_force": ("导丝按受力变色（红→黄 = 0→1 N）", "Colour wire by force (red→yellow = 0→1 N)"),
    "xray": ("X 光", "X-ray"),
    "xray_on": ("X 光模式", "X-ray mode"),
    "xray_hint": ("沿 C 臂方向看，近似平行投影；不打造影剂时只看得到器械",
                  "Looks along the C-arm, nearly parallel; without contrast only the instruments show"),
    "xray_contrast": ("打造影剂（路图）：显示血管", "Contrast (road map): show the vessels"),
    "xray_policy": ("标出策略拿到的东西（跟踪点、目标）", "Mark what the policy gets (tracked points, target)"),
    "view": ("视角", "View"),
    "view_fluoro": ("透视机方向", "Fluoroscope"),
    "view_side": ("侧面", "Side"),
    "view_tip": ("尖端附近", "Near the tip"),
    "follow": ("相机跟着尖端走", "Camera follows the tip"),
    "save": ("保存", "Save"),
    "save_episode": ("保存这一回合", "Save this episode"),
    "save_to": ("存到 {path}", "Saves to {path}"),
    "saved": ("已保存：{path}", "Saved: {path}"),
    "save_failed": ("保存失败：{error}", "Save failed: {error}"),
    # run state
    "state_paused": ("暂停", "Paused"),
    "state_running": ("运行中", "Running"),
    "state_reviewing": ("回看中", "Reviewing"),
    "state_loading": ("正在加载策略…", "Loading the policy…"),
    "state_load_failed": ("策略加载失败：{error}", "Could not load the policy: {error}"),
    "state_switching": ("正在打开环境…", "Opening the environment…"),
    "state_switch_failed": ("环境打不开：{error}", "Could not open the environment: {error}"),
    "state_view_only": ("只看形状", "View only"),
    # status table
    "status": ("状态", "Status"),
    "episode_step": ("回合 · 步", "Episode · step"),
    "episode_step_value": ("第 {episode} 回合 · 第 {step} 步", "episode {episode} · step {step}"),
    "step_limit": ("（上限 {limit}）", " (limit {limit})"),
    "sim_time": ("仿真时间", "Simulated time"),
    "step_reward": ("这一步奖励", "Reward this step"),
    "return": ("累计奖励", "Return so far"),
    "tip_to_target": ("尖端到目标（直线）", "Tip to target (straight line)"),
    "reach_within": ("（到达 ≤ {mm} mm）", " (reached at ≤ {mm} mm)"),
    "inserted_rotation": ("送入长度 · 转角", "Inserted length · rotation"),
    "inst_name_guidewire": ("导丝", "guidewire"),
    "inst_name_catheter": ("导管", "catheter"),
    "seed": ("种子", "Seed"),
    "scene": ("场景", "Scene"),
    "error": ("错误", "Error"),
    "outcome_reached": ("到达目标", "target reached"),
    "outcome_error": ("仿真出错", "simulation error"),
    "outcome_truncated": ("到步数上限，没到", "step limit, target not reached"),
    "outcome_running": ("进行中", "in progress"),
    # sidebar panels
    "no_action": ("还没有动作（刚重置）", "No action yet (just reset)"),
    "action_n": ("动作 {i}", "action {i}"),
    "nominal": ("（原本 {value}）", " (before filter {value})"),
    "no_forces": ("还没有受力数据", "No contact force yet"),
    "measured": ("实测", "measured"),
    "predicted": ("模型预测", "predicted"),
    # instrument window
    "instrument_heading": ("**器械不受力时的样子**", "**The instruments at rest**"),
    "instrument_distal": ("末端 {mm} mm，按比例；线宽 = 粗细", "Distal {mm} mm to scale; line width = diameter"),
    "instrument_whole": ("全长 {mm} mm：灰 = 杆身，红 = 软段", "Whole length {mm} mm: grey = shaft, red = soft section"),
    "inst_kind": ("类型", "Type"),
    "inst_length": ("全长", "Length"),
    "inst_mass": ("质量", "Mass"),
    "inst_bend": ("弯头", "Bend"),
    "inst_bend_value": ("{deg}°，半径 {r} mm", "{deg}°, radius {r} mm"),
    "inst_length_short": ("长度", "length"),
    "inst_diameter": ("外径", "outer d."),
    "inst_density": ("密度", "density"),
    "inst_shaft": ("杆身", "shaft"),
    "inst_tip": ("软段", "soft section"),
    "inst_units_hint": ("E = 杨氏模量。密度是仿真里填的数（镍钛约 6.45 g/cm³）",
                        "E = Young's modulus. Density as the simulator has it (nitinol is about 6.45 g/cm³)"),
    "no_instruments": ("这个环境里没有器械（只看形状）", "No instruments here (view only)"),
}

# built-in policy labels (they double as menu keys and as --policy values)
POLICY_TEXT: Dict[str, Pair] = {
    "随机动作": ("随机动作", "random actions"),
    "不动": ("不动", "hold still"),
    "只往前推": ("只往前推", "push forward only"),
    "键盘手动": ("键盘手动", "keyboard (by hand)"),
    "手写推法": ("手写推法", "scripted procedure"),
    "不过滤": ("不过滤", "no filter"),
    "回放": ("回放", "replay"),
    "（未选）": ("（未选）", "(none)"),
}

_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def pick(pair: Pair, lang: str) -> str:
    return pair[0] if lang == "zh" else pair[1]


def tr(key: str, lang: str, **values) -> str:
    text = pick(TEXT[key], lang)
    return text.format(**values) if values else text


def other(lang: str) -> str:
    return LANGS[(LANGS.index(lang) + 1) % len(LANGS)]


def policy_text(label: str, lang: str) -> str:
    """A policy's label in ``lang``; built-in names are translated, checkpoint names kept."""
    for name, pair in POLICY_TEXT.items():
        label = label.replace(name, pick(pair, lang))
    return label


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))
