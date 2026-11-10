"""导丝工作台：选 phantom 版本和仿真设置、改参数，部署策略，看器械、中心线和 X 光画面。

    conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py
    conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py \\
        --phantom spheres_cut --setting phantom_friction --checkpoint runs/<run>/checkpoints/step_N.pt --play
    conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py \\
        --replay runs/workbench/<file>.npz            # 回放保存的一回合，不需要 SOFA

然后用 Chrome 打开 http://localhost:8080 。界面默认中文，`--lang en` 起英文；面板最上面的按钮一键切换。

The simulator runs in the main thread; GUI callbacks (viser's thread) only set flags that
the loop picks up, so they never touch SOFA. Switching the language, the phantom, the
setting or applying parameters rebuilds the whole panel (viser cannot relabel tabs or swap
dropdown menus), keeping every display setting. Display-only switches (layers, X-ray) act
directly under the lock.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # this checkout: runs/, recordings
CODE = Path(os.environ.get("WORKBENCH_CODE") or ROOT).resolve()   # simulator code and configs (parts.environments)
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(CODE))                            # guidewire_rl, simulator_modifications from there

from parts import keyboard, parameters, scripted  # noqa: E402
from parts.body import BodyLayer  # noqa: E402
from parts.environments import (DEFAULT_FRICTION, DEFAULT_PHANTOM, DEFAULT_SETTING, FRICTIONS, Phantom,  # noqa: E402
                                Setting, build_config, compatible, find_setting, friction_of, locate, phantoms,
                                run_config_of, settings)
from parts.frame import EpisodeLog, Frame  # noqa: E402
from parts.i18n import LANGS, other, pick, policy_text, tr  # noqa: E402
from parts.instrument import instruments_html  # noqa: E402
from parts.plots import PlotPanel  # noqa: E402
from parts.policies import (CheckpointPolicy, Policy, checkpoint_label, discover_checkpoints,  # noqa: E402
                            policy_menu)
from parts.scene import (CenterlineLayer, ForceArrowLayer, RadiusTubeLayer, TargetLayer,  # noqa: E402
                         TrackedPointsLayer, TrailLayer, VesselLayer, WireLayer, apply_camera, camera_pose)
from parts.text import action_html, force_html, status_html  # noqa: E402
from parts.xray import CONTRAST_LOOK, LIGHT, WIRE_DARK, XrayMode, visible_in_xray  # noqa: E402

VIEWS = ("fluoro", "side", "tip")   # camera presets; button text is tr("view_<preset>")
LAYERS = ("vessel", "body", "centerlines", "other_centerlines", "radius_tubes", "wire", "trail", "target",
          "tracked", "force")
LAYER_DEFAULTS = {"other_centerlines": False, "radius_tubes": False, "tracked": False}
Menu = Dict[str, Callable[[], Policy]]
Configure = Callable[[Phantom, Setting], Dict]          # (phantom, setting) -> environment config
Opener = Callable[[Phantom, Setting, Dict], Tuple[object, Menu]]   # ... + final config -> (source, menu)


class Workbench:
    def __init__(self, source, *, menu: Menu, initial_policy: Optional[str], port: int, record_dir: Path,
                 seed: int, live: bool, play: bool = False, speed: float = 7.5, lang: str = "zh",
                 phantom_list: Sequence[Phantom] = (), setting_list: Sequence[Setting] = (),
                 phantom_key: Optional[str] = None, setting_key: Optional[str] = None,
                 friction: Optional[float] = None,
                 configure: Optional[Configure] = None, opener: Optional[Opener] = None) -> None:
        import viser

        self.source, self.menu = source, menu
        self.live = live and (source is None or getattr(source, "can_run", True))
        self.key_server = keyboard.KeyServer(keyboard.KEY_STATE) if self.live else None
        self.phantoms, self.settings_list = list(phantom_list), list(setting_list)
        self.phantom_key, self.setting_key = phantom_key, setting_key
        self.friction = friction                     # chosen apart from the setting; None = the setting's own
        self.configure, self.opener = configure, opener
        self.param_base: Optional[Dict] = None      # the setting's own values (stored units)
        self.param_values: Optional[Dict] = None    # the values in effect
        self.record_dir, self.next_seed = Path(record_dir), int(seed)
        self._initial_speed = float(min(max(speed, 1.0), 30.0))
        self.lang = lang
        self.lock = threading.RLock()
        self.server = viser.ViserServer(port=port, label=tr("title", lang))
        self.server.gui.configure_theme(dark_mode=True, control_width="large", show_share_button=False,
                                        brand_color=(29, 158, 117))
        self.server.gui.main_panel.dock_right()
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_environment_map(LIGHT[0], environment_intensity=LIGHT[1])

        # what the GUI asks the loop to do
        self.playing = bool(play) and (self.live or bool(getattr(source, "can_run", False)))   # --play: replays too
        self.step_once = False
        self.reset_requested = False
        self.pending_policy: Optional[str] = initial_policy if self.live else None
        self.pending_env: Optional[Tuple[str, str, Optional[Dict]]] = None   # phantom, setting, parameters
        self.env_failed = False                                   # the last switch did not open
        self.rebuild_requested: Optional[str] = None     # "language" | "environment"
        self.view_index: Optional[int] = None   # None = follow the newest frame
        self.policy_label = (initial_policy or "（未选）") if live else "回放"   # shown through policy_text()
        self.state, self.state_detail = "paused", ""             # shown as tr("state_<state>")

        self.log: Optional[EpisodeLog] = None
        self.frame: Optional[Frame] = None
        self._vessel_key = None
        self._last_tip = None
        self.xray = XrayMode(self.server)
        self._build_scene()
        self._build_gui()

        @self.server.on_client_connect
        def _(client) -> None:
            with self.lock:
                if self.log is None:
                    return
                if self.xray.on:
                    self.xray.apply_camera(client, self.log.info)
                else:
                    apply_camera([client], camera_pose("fluoro", self.log.info))

    # ------------------------------------------------------------------ build --
    def _build_scene(self) -> None:
        s = self.server
        self.layers = {
            "vessel": VesselLayer(s), "body": BodyLayer(s), "centerlines": CenterlineLayer(s),
            "other_centerlines": CenterlineLayer(s, "/other_centerlines", task=False),
            "radius_tubes": RadiusTubeLayer(s), "wire": WireLayer(s), "trail": TrailLayer(s),
            "target": TargetLayer(s), "tracked": TrackedPointsLayer(s), "force": ForceArrowLayer(s),
        }

    def _settings(self) -> Dict:
        """Everything the panel holds, to carry over a rebuild."""
        return {"policy": self._policy_keys.get(self.policy_dd.value), "speed": float(self.speed.value),
                "auto_next": bool(self.auto_next.value), "seed": int(self.seed_in.value),
                "layers": {k: bool(cb.value) for k, cb in self.layer_boxes.items()},
                "opacity": float(self.opacity.value), "body_opacity": float(self.body_opacity.value),
                "color_force": bool(self.color_force.value),
                "follow": bool(self.follow.value), "xray": bool(self.xray_box.value),
                "key_speed": float(self.key_speed.value),
                "contrast": bool(self.contrast_box.value), "policy_marks": bool(self.policy_box.value)}

    def _build_gui(self, keep: Optional[Dict] = None) -> None:
        """Build the side panel in ``self.lang``; ``keep`` carries the settings over a rebuild."""
        import viser

        keep = keep or {}
        lang, gui = self.lang, self.server.gui
        t = lambda key, **values: tr(key, lang, **values)
        switch = gui.add_button(t("switch_to"), icon=viser.Icon.LANGUAGE)
        switch.on_click(lambda _: self._ask_rebuild("language"))
        self.status = gui.add_html(f"<div style='opacity:.6'>{t('starting')}</div>")

        if self.phantoms and self.settings_list:
            self._build_environment_folder(gui, t)

        with gui.add_folder(t("run")):
            keys = list(self.menu) or [self.policy_label]     # replay: 回放; view only: （未选）
            self._policy_keys = {policy_text(k, lang): k for k in keys}
            current = keep.get("policy") or self.pending_policy or self.policy_label
            current = current if current in keys else keys[0]
            self.policy_dd = gui.add_dropdown(t("policy"), options=list(self._policy_keys),
                                              initial_value=policy_text(current, lang), disabled=not self.live)
            self.policy_dd.on_update(lambda _: self._ask_policy(self.policy_dd.value))
            self.keys_html = gui.add_html("")
            self.key_speed = gui.add_slider(t("keys_speed"), min=0.05, max=1.0, step=0.05,
                                            initial_value=keep.get("key_speed", keyboard.KEY_STATE.speed))
            self.key_speed.on_update(lambda _: setattr(keyboard.KEY_STATE, "speed", float(self.key_speed.value)))
            self._refresh_keys()
            # a replay plays without a policy; a view-only shape does not play at all
            can_play = bool(getattr(self.source, "can_run", False)) and (self.live or not self.menu)
            self.play_btn = gui.add_button(t("pause") if self.playing else t("play"),
                                           icon=viser.Icon.PLAYER_PLAY, color="green", disabled=not can_play)
            self.play_btn.on_click(lambda _: self._toggle_play())
            self._row = {t("step"): "step", t("restart"): "restart"}
            row = gui.add_button_group("", options=list(self._row), disabled=not can_play)
            row.on_click(lambda e: self._on_row(self._row[e.target.value]))
            self.speed = gui.add_slider(t("speed"), min=1.0, max=100.0, step=0.5,
                                        initial_value=keep.get("speed", self._initial_speed), hint=t("speed_hint"))
            self.auto_next = gui.add_checkbox(t("auto_next"), initial_value=keep.get("auto_next", True))
            self.seed_in = gui.add_number(t("next_seed"), initial_value=keep.get("seed", self.next_seed), step=1,
                                          hint=t("next_seed_hint"))
            self.scrub = gui.add_slider(t("scrub"), min=0, max=1, step=1, initial_value=0, hint=t("scrub_hint"))
            self.scrub.on_update(lambda _: self._on_scrub())
            gui.add_button(t("latest")).on_click(lambda _: self._follow_latest())

        tabs = gui.add_tab_group()
        with tabs.add_tab(t("tab_curves"), icon=viser.Icon.CHART_LINE):
            gui.add_markdown(t("action_heading"))
            self.action_html = gui.add_html("")
            gui.add_markdown(t("force_heading"))
            self.force_html = gui.add_html("")
            self.plots = PlotPanel(self.server, lang=lang)
        with tabs.add_tab(t("tab_instrument"), icon=viser.Icon.RULER_MEASURE):
            gui.add_markdown(t("instrument_heading"))
            self.instrument_html = gui.add_html("")
        with tabs.add_tab(t("parameters"), icon=viser.Icon.ADJUSTMENTS):
            self._build_parameter_tab(gui, t)
        with tabs.add_tab(t("tab_display"), icon=viser.Icon.SETTINGS):
            with gui.add_folder(t("xray")):
                gui.add_markdown(f"<small>{t('xray_hint')}</small>")
                self.xray_box = gui.add_checkbox(t("xray_on"), initial_value=keep.get("xray", False))
                self.contrast_box = gui.add_checkbox(t("xray_contrast"), initial_value=keep.get("contrast", False))
                self.policy_box = gui.add_checkbox(t("xray_policy"), initial_value=keep.get("policy_marks", True))
                for box in (self.xray_box, self.contrast_box, self.policy_box):
                    box.on_update(lambda _: self._apply_display())
            with gui.add_folder(t("layers")):
                self.layer_boxes = {}
                for key in LAYERS:
                    initial = keep.get("layers", {}).get(key, LAYER_DEFAULTS.get(key, True))
                    cb = gui.add_checkbox(t(f"layer_{key}"), initial_value=initial)
                    cb.on_update(lambda _: self._apply_display())
                    self.layer_boxes[key] = cb
                self.opacity = gui.add_slider(t("opacity"), min=0.05, max=1.0, step=0.05,
                                              initial_value=keep.get("opacity", 0.35))
                self.opacity.on_update(lambda _: self._layer("vessel", "set_opacity", self.opacity.value))
                self.body_opacity = gui.add_slider(t("body_opacity"), min=0.05, max=1.0, step=0.05,
                                                   initial_value=keep.get("body_opacity", 0.15))
                self.body_opacity.on_update(lambda _: self._layer("body", "set_opacity", self.body_opacity.value))
                self.color_force = gui.add_checkbox(t("color_force"), initial_value=keep.get("color_force", True))
            with gui.add_folder(t("view")):
                self._views = {t(f"view_{preset}"): preset for preset in VIEWS}
                views = gui.add_button_group("", options=list(self._views))
                views.on_click(lambda e: self._set_view(self._views[e.target.value]))
                self.follow = gui.add_checkbox(t("follow"), initial_value=keep.get("follow", False))
            with gui.add_folder(t("save")):
                gui.add_button(t("save_episode"), icon=viser.Icon.DEVICE_FLOPPY).on_click(lambda _: self._save())
                self.save_status = gui.add_html(f"<small>{t('save_to', path=self.record_dir)}</small>")

    def _phantom(self, key: Optional[str] = None) -> Optional[Phantom]:
        key = key or self.phantom_key
        return next((p for p in self.phantoms if p.key == key), None)

    def _setting(self, key: Optional[str] = None) -> Optional[Setting]:
        return find_setting(key or self.setting_key or "", self.settings_list)

    def _build_environment_folder(self, gui, t) -> None:
        lang = self.lang
        phantom, setting = self._phantom() or self.phantoms[0], self._setting() or self.settings_list[0]
        with gui.add_folder(t("environment")):
            self._phantom_keys = {pick(p.label, lang): p.key for p in self.phantoms}
            self.phantom_dd = gui.add_dropdown(t("phantom_pick"), options=list(self._phantom_keys),
                                               initial_value=pick(phantom.label, lang), hint=t("env_hint"))
            self.phantom_dd.on_update(lambda _: self._ask_env(phantom=self._phantom_keys[self.phantom_dd.value]))
            extra = "" if phantom.kind == "sim" else f"<br><b>{t('shape_only')}</b>"
            gui.add_html(f"<small style='opacity:.7'>{pick(phantom.note, lang)}{extra}</small>")
            self._setting_keys = {pick(s.label, lang): s.key for s in self.settings_list}
            self.setting_dd = gui.add_dropdown(t("setting_pick"), options=list(self._setting_keys),
                                               initial_value=pick(setting.label, lang), hint=t("setting_hint"))
            self.setting_dd.on_update(lambda _: self._ask_env(setting=self._setting_keys[self.setting_dd.value]))
            also = (f"<br>{t('setting_also', names=', '.join(setting.also))}" if setting.also else "")
            gui.add_html(f"<small style='opacity:.7'>{setting.config_path.name}: {pick(setting.note, lang)}"
                         f"{also}</small>")
            current = self.friction if self.friction is not None else (
                self.param_values or {}).get("friction", DEFAULT_FRICTION)
            choices = dict(FRICTIONS)
            if not any(abs(current - f) < 1e-12 for f in choices):
                choices[current] = (f"{current:g}（参数页改的）", f"{current:g} (set in Parameters)")
            self._friction_keys = {pick(label, lang): f for f, label in choices.items()}
            shown = next(pick(label, lang) for f, label in choices.items() if abs(current - f) < 1e-12)
            self.friction_dd = gui.add_dropdown(t("friction_pick"), options=list(self._friction_keys),
                                                initial_value=shown, hint=t("friction_hint"))
            self.friction_dd.on_update(lambda _: self._ask_friction(self._friction_keys[self.friction_dd.value]))

    def _build_parameter_tab(self, gui, t) -> None:
        """The simulation numbers of the open environment, editable; applying rebuilds it."""
        self.param_inputs = {}
        if self.param_values is None:
            gui.add_markdown(f"<small>{t('no_params')}</small>")
            return
        lang, phantom = self.lang, self._phantom()
        edited = set(parameters.changed(self.param_base, self.param_values))
        for p in parameters.PARAMS:
            label = pick(p.label, lang) + (f" ({p.unit})" if p.unit else "") + (t("edited") if p.key in edited else "")
            value = parameters.shown(p, self.param_values[p.key])
            disabled = phantom is not None and phantom.kind == "shape" and not p.view_only_too
            if p.vector:
                handle = gui.add_vector2(label, initial_value=value, step=p.step, disabled=disabled)
            else:
                handle = gui.add_number(label, initial_value=value, step=p.step, disabled=disabled)
            self.param_inputs[p.key] = handle
        gui.add_button(t("apply_params"), color="green").on_click(lambda _: self._ask_params(apply=True))
        gui.add_button(t("reset_params")).on_click(lambda _: self._ask_params(apply=False))
        gui.add_markdown(f"<small>{t('params_hint')}</small>")

    def _rebuild_gui(self) -> None:
        keep = self._settings()
        self.server.gui.reset()
        self.server.gui.main_panel.dock_right()   # reset() undocks the panel
        self.server.gui.set_panel_label(tr("title", self.lang))
        self._build_gui(keep)
        if self.log is not None and self.frame is not None:
            index = len(self.log) - 1 if self.view_index is None else self.view_index
            self._show(self.log.frames[index], index)
            self.scrub.value = index
        else:
            self._refresh_status()

    # -------------------------------------------------------------- callbacks --
    def _layer(self, key: str, method: str, value) -> None:
        with self.lock:
            getattr(self.layers[key], method)(value)

    def _ask_policy(self, shown: str) -> None:
        self.pending_policy = self._policy_keys.get(shown, shown)

    def _ask_env(self, phantom: Optional[str] = None, setting: Optional[str] = None) -> None:
        """A new phantom keeps edited parameters; a new setting starts from its own values. After a
        failed switch any pick tries again, keeping the other half of what was asked for."""
        if phantom is not None and (phantom != self.phantom_key or self.env_failed):
            self.pending_env = (phantom, self.setting_key, self.param_values)
        elif setting is not None and (setting != self.setting_key or self.env_failed):
            self.pending_env = (self.phantom_key, setting, None)

    def _ask_params(self, apply: bool) -> None:
        if not apply:
            self.pending_env = (self.phantom_key, self.setting_key, None)
            return
        values = {p.key: parameters.stored(p, self.param_inputs[p.key].value) for p in parameters.PARAMS}
        self.friction = float(values["friction"])      # the friction menu follows the parameter
        self.pending_env = (self.phantom_key, self.setting_key, values)

    def _ask_friction(self, friction: float) -> None:
        """Another wall friction: the same environment rebuilt, other edited parameters kept."""
        if self.friction is not None and abs(friction - self.friction) < 1e-12 and not self.env_failed:
            return
        self.friction = float(friction)
        values = None if self.param_values is None else dict(self.param_values, friction=self.friction)
        self.pending_env = (self.phantom_key, self.setting_key, values)

    def _ask_rebuild(self, why: str) -> None:
        self.rebuild_requested = why

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.view_index = None

    def _on_row(self, which: str) -> None:
        if which == "step":
            self.playing, self.step_once, self.view_index = False, True, None
        else:
            self.reset_requested = True

    def _on_scrub(self) -> None:
        if self.log is not None and int(self.scrub.value) < len(self.log) - 1:
            self.playing = False
            self.view_index = int(self.scrub.value)

    def _follow_latest(self) -> None:
        self.view_index = None

    def _set_view(self, preset: str) -> None:
        with self.lock:
            if self.log is None:
                return
            if self.xray_box.value:
                self.xray_box.value = False      # a normal view leaves X-ray mode
                self._apply_display()
            tip = None if self.frame is None else self.frame.tip_mm
            apply_camera(list(self.server.get_clients().values()), camera_pose(preset, self.log.info, tip))

    def _apply_display(self) -> None:
        """Layer checkboxes and the X-ray switches -> what the scene shows."""
        with self.lock:
            clients = list(self.server.get_clients().values())
            info = None if self.log is None else self.log.info
            if self.xray_box.value and not self.xray.on and info is not None:
                self.xray.enter(info, clients)
            elif not self.xray_box.value and self.xray.on:
                self.xray.leave(clients)
                if info is not None:
                    apply_camera(clients, camera_pose("fluoro", info))
            contrast, marks = bool(self.contrast_box.value), bool(self.policy_box.value)
            for key, layer in self.layers.items():
                shown = bool(self.layer_boxes[key].value)
                if self.xray.on:
                    rule = visible_in_xray(key, contrast, marks)
                    shown = shown if rule is None else rule
                layer.set_visible(shown)
            self.layers["vessel"].set_look(CONTRAST_LOOK if self.xray.on else None)
            self.layers["wire"].color_override = WIRE_DARK if self.xray.on else None
            if self.frame is not None:
                self._draw_wire(self.frame)

    def _save(self) -> None:
        with self.lock:
            if self.log is None or not len(self.log):
                return
            f = self.log.frames[-1]
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_ep{f.episode}_seed{f.seed}.npz"
            try:
                path = self.log.save(self.record_dir / name)
                self.save_status.content = f"<small>{tr('saved', self.lang, path=path)}</small>"
            except Exception as exc:  # disk or format problem: show it, keep running
                self.save_status.content = (f"<small style='color:#E24B4A'>"
                                            f"{tr('save_failed', self.lang, error=repr(exc))}</small>")

    # ------------------------------------------------------------------- loop --
    def _reset(self) -> None:
        seed = int(self.seed_in.value)
        frame = self.source.reset(seed=seed)
        if self.live:
            self.seed_in.value = seed + 1
        self.log = EpisodeLog(self.source.scene_info())
        self.log.append(frame)
        self.view_index = None
        self._show(frame, 0)

    def _load_policy(self, label: str) -> None:
        self.state = "loading"
        self._refresh_status()
        try:
            policy = self.menu[label]()
        except Exception as exc:  # bad checkpoint etc.: keep the old policy
            self.state, self.state_detail = "load_failed", repr(exc)[:200]
            self._refresh_status()
            return
        self.source.set_policy(policy)
        self.policy_label = policy.label
        self.state = "paused"
        self._refresh_keys()
        self._reset()

    def _refresh_keys(self) -> None:
        """The keyboard script and its speed slider: active only while the keyboard drives."""
        active = self.live and self.policy_label == keyboard.LABEL
        two = int(getattr(self.source, "action_dim", 2) or 2) >= 4
        port = None if self.key_server is None else self.key_server.port
        self.keys_html.content = keyboard.capture_html(port, active, two, self.lang)
        self.key_speed.visible = active

    def _open_env(self, phantom_key: str, setting_key: str, values: Optional[Dict]) -> None:
        phantom, setting = self._phantom(phantom_key), self._setting(setting_key)
        self.playing = False
        self.state, self.state_detail = "switching", ""
        self._refresh_status()
        old, self.source = self.source, None
        if old is not None:
            old.close()
        try:
            config = self.configure(phantom, setting)
            if self.friction is not None:                  # chosen apart from the setting
                config = parameters.apply(config, {"friction": self.friction})
            base = parameters.read(config)
            values = base if values is None else values
            if parameters.changed(base, values):
                config = parameters.apply(config, values)
            source, menu = self.opener(phantom, setting, config)
        except Exception as exc:  # broken config, SOFA failure: say so, keep the panel
            self.state, self.state_detail = "switch_failed", repr(exc)[:300]
            self.log = self.frame = None
            # the panel shows what was asked for; the next pick (even the same) tries again with it
            self.phantom_key, self.setting_key, self.env_failed = phantom_key, setting_key, True
            self._refresh_status()
            return
        edited = parameters.changed(base, values)
        if edited:                                  # say so in the status table's scene row
            note = " · ✎ " + ", ".join(edited)
            if hasattr(source, "description"):
                source.description += note
            if getattr(source, "_info", None) is not None:
                source._info.description += note
        self.source, self.menu = source, menu
        self.phantom_key, self.setting_key, self.env_failed = phantom.key, setting.key, False
        self.param_base, self.param_values = base, values
        self.live = bool(getattr(source, "can_run", True)) and bool(menu)
        label = self.policy_label if self.policy_label in menu else next(iter(menu), "（未选）")
        self.policy_label = label
        self._kept["policy"] = label
        self._vessel_key = None
        self.log = self.frame = None
        self.server.gui.reset()
        self.server.gui.main_panel.dock_right()
        self.server.gui.set_panel_label(tr("title", self.lang))
        self._build_gui(self._kept)
        if self.live:
            self._load_policy(label)
        else:
            self.state = "view_only"
            self._reset()

    def run(self) -> None:
        print(tr("open_url", self.lang, url=f"http://localhost:{self.server.get_port()}"), flush=True)
        last_step = 0.0
        with self.lock:
            if self.source is None and self.phantom_key is not None:
                self._kept = self._settings()
                self._open_env(self.phantom_key, self.setting_key, None)
            elif self.live:
                self._load_policy(self.pending_policy)
            else:
                self._reset()
            self.pending_policy = None
        while True:
            time.sleep(0.005)
            with self.lock:
                if self.rebuild_requested is not None:
                    if self.rebuild_requested == "language":
                        self.lang = other(self.lang)
                    self.rebuild_requested = None
                    self._rebuild_gui()
                    continue
                if self.pending_env is not None:
                    request, self.pending_env = self.pending_env, None
                    self._kept = self._settings()
                    self._open_env(*request)
                    continue
                if self.source is None or self.log is None:
                    continue
                if self.pending_policy is not None:
                    label, self.pending_policy = self.pending_policy, None
                    if self.live:
                        self._load_policy(label)
                    continue
                if self.reset_requested:
                    self.reset_requested = False
                    self._reset()
                    continue
                if self.view_index is not None:
                    shown = self.log.frames[self.view_index]
                    if shown is not self.frame:
                        self._show(shown, self.view_index)
                    continue
                due = self.playing and time.time() - last_step >= 1.0 / float(self.speed.value)
                if not (due or self.step_once):
                    self._sync_play_button()
                    continue
                self.step_once = False
                last_step = time.time()
                if self.log.frames[-1].done:
                    if self.playing and self.auto_next.value:
                        self._reset()
                    else:
                        self.playing = False
                    continue
                self.state = "running" if self.playing else "paused"
                frame = self.source.step()
                self.log.append(frame)
                self._show(frame, len(self.log) - 1)

    # ----------------------------------------------------------------- render --
    def _sync_play_button(self) -> None:
        label = tr("pause" if self.playing else "play", self.lang)
        if self.play_btn.label != label:
            self.play_btn.label = label
        if self.state in ("loading", "load_failed", "switching", "switch_failed", "view_only"):
            return
        state = "running" if self.playing else ("reviewing" if self.view_index is not None else "paused")
        if state != self.state:
            self.state = state
            self._refresh_status()

    def _refresh_status(self) -> None:
        state = tr(f"state_{self.state}", self.lang, error=self.state_detail)
        if self.log is not None and self.frame is not None:
            self.status.content = status_html(self.log.info, self.frame, policy_text(self.policy_label, self.lang),
                                              state, self.lang)
        else:
            self.status.content = f"<div>{state}</div>"

    def _draw_wire(self, frame: Frame) -> None:
        force = frame.labels.get("distal_force") if self.color_force.value else None
        instruments = self.log.info.instruments if self.log is not None else None
        self.layers["wire"].update(frame, force, instruments)

    def _show(self, frame: Frame, index: int) -> None:
        info, log = self.log.info, self.log
        self.frame = frame
        if info.vessel_key != self._vessel_key:
            self._vessel_key = info.vessel_key
            for key in ("vessel", "centerlines", "other_centerlines", "radius_tubes", "target", "body"):
                self.layers[key].set_scene(info)
            clients = list(self.server.get_clients().values())
            if self.xray.on:
                self.xray.enter(info, clients)
            else:
                apply_camera(clients, camera_pose("fluoro", info))
            self.instrument_html.content = instruments_html(info.instruments, self.lang)
            self._apply_display()
        with self.server.atomic():
            self._draw_wire(frame)
            self.layers["trail"].update(np.array([f.tip_mm for f in log.frames[: index + 1]
                                                  if f.tip_mm is not None]).reshape(-1, 3))
            self.layers["target"].update(frame)
            self.layers["tracked"].update(frame, info)
            self.layers["force"].update(frame)
        if self.follow.value and frame.tip_mm is not None:
            self._follow_tip(frame.tip_mm)
        self._last_tip = frame.tip_mm
        nominal = frame.extras.get("nominal_action")
        lang = self.lang
        if not self.instrument_html.content:
            self.instrument_html.content = instruments_html(info.instruments, lang)
        self.action_html.content = action_html(info, frame, None if nominal is None else np.asarray(nominal), lang)
        self.force_html.content = force_html(frame, lang)
        self.plots.update(log, index)
        self.scrub.max = max(1, len(log) - 1)
        if self.view_index is None:
            self.scrub.value = len(log) - 1
        self._refresh_status()

    def _follow_tip(self, tip_mm: np.ndarray) -> None:
        if self._last_tip is None:
            return
        shift = (np.asarray(tip_mm) - np.asarray(self._last_tip)) * 1e-3
        for client in self.server.get_clients().values():
            try:
                with client.atomic():
                    client.camera.position = tuple(np.asarray(client.camera.position) + shift)
                    client.camera.look_at = tuple(np.asarray(client.camera.look_at) + shift)
            except AssertionError:  # client has not reported its camera yet
                pass


# ------------------------------------------------------------------- opening --
def make_opener(args) -> Tuple[Configure, Opener]:
    """(phantom, setting) -> config, and (phantom, setting, config) -> (source, policy menu).
    Checkpoints are listed when their run had the same observation and action as the
    environment (plus every --checkpoint, whatever it is)."""
    from parts.sources import LiveEnvSource, ShapeSource

    explicit = [Path(p) for p in args.checkpoint]

    def configure(phantom: Phantom, setting: Setting) -> Dict:
        return build_config(phantom, setting, args.set)

    def opener(phantom: Phantom, setting: Setting, config: Dict):
        name = f"{phantom.key} · {setting.key}"
        if phantom.kind == "shape":
            angle = parameters.read(config)["carm"]
            return ShapeSource(phantom.shape["mesh"], phantom.shape["centerlines"], angle, description=name), {}
        source = LiveEnvSource(config, description=name)
        found = [p for p in discover_checkpoints(args.runs)
                 if p.resolve() not in {c.resolve() for c in explicit}]
        fitting = [p for p in found if (rc := run_config_of(p)) is not None and compatible(config, rc)]

        def make(path: Path) -> Policy:
            run_config = run_config_of(path) or config
            return CheckpointPolicy(run_config, path, observation_dim=source.observation_dim,
                                    action_dim=source.action_dim, episode_length=source.env.max_episode_steps,
                                    device=args.device)
        by_hand = {keyboard.LABEL: lambda: keyboard.KeyboardPolicy(source.action_dim, keyboard.KEY_STATE)}
        staged = scripted.staged_factory(source)
        if staged is not None:
            by_hand[scripted.LABEL] = staged
        return source, policy_menu(source.action_dim, explicit + fitting, make, extra=by_hand)
    return configure, opener


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", "--config", dest="env", default=None,
                        help="open the phantom x setting of this config (a name or path in configs/)")
    parser.add_argument("--phantom", help="phantom version to open (a key of PHANTOMS); overrides --env's")
    parser.add_argument("--setting", help="setting to open (a config name); overrides --env's")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value (every environment opened)")
    parser.add_argument("--checkpoint", action="append", default=[], help="add a checkpoint to the policy menu")
    parser.add_argument("--policy", help="menu entry to start with (default: the first --checkpoint, else 随机动作)")
    parser.add_argument("--replay", help="play a saved episode (.npz) instead of running the simulator")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="where to look for checkpoints")
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first episode")
    parser.add_argument("--device", help="torch device for checkpoints (default: the config's)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--play", action="store_true", help="start playing right away")
    parser.add_argument("--speed", type=float, default=7.5, help="steps per second to start with (1-100)")
    parser.add_argument("--record-dir", default=str(ROOT / "runs" / "workbench"))
    parser.add_argument("--lang", choices=LANGS, default="zh", help="interface language to start with")
    args = parser.parse_args()

    if args.replay:
        from parts.sources import ReplaySource
        bench = Workbench(ReplaySource(args.replay), menu={}, initial_policy=None, port=args.port,
                          record_dir=Path(args.record_dir), seed=args.seed, live=False, play=args.play,
                          speed=args.speed, lang=args.lang)
    else:
        phantom_list, setting_list = phantoms(), settings()
        phantom_key = setting_key = None
        friction = DEFAULT_FRICTION
        if args.env:
            env = Path(args.env) if Path(args.env).suffix else CODE / "configs" / f"{args.env}.yaml"
            phantom_key, setting_key = locate(env, phantom_list, setting_list)
            friction = friction_of(env)
        if args.setting:                              # a config name: its setting, and its friction
            found = find_setting(args.setting, setting_list)
            setting_key = found.key if found is not None else args.setting
            path = CODE / "configs" / f"{args.setting}.yaml"
            friction = friction_of(path) if path.exists() else friction
        phantom_key = args.phantom or phantom_key or DEFAULT_PHANTOM
        setting_key = setting_key or DEFAULT_SETTING
        configure, opener = make_opener(args)
        initial = args.policy or (checkpoint_label(Path(args.checkpoint[0])) if args.checkpoint else None)
        bench = Workbench(None, menu={}, initial_policy=initial or "随机动作", port=args.port,
                          record_dir=Path(args.record_dir), seed=args.seed, live=True, play=args.play,
                          speed=args.speed, lang=args.lang, phantom_list=phantom_list, setting_list=setting_list,
                          phantom_key=phantom_key, setting_key=setting_key, friction=friction,
                          configure=configure, opener=opener)
    try:
        bench.run()
    except KeyboardInterrupt:
        pass
    finally:
        if bench.source is not None:
            bench.source.close()


if __name__ == "__main__":
    main()
