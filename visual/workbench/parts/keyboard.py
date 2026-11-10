"""Driving the instruments by hand from the keyboard ("键盘手动" in the policy menu).

Keys (held = moving, at ``KeyState.speed`` of the top speed; released = still):
    guidewire  ↑ / ↓ push / pull,  ← / → rotate
    catheter   W / S push / pull,  A / D rotate     (only with a second instrument)

viser hands no key events to Python, so the panel carries a small script
(``capture_html``, run from an image's ``onerror``) that listens to the keys in the
browser and reports the held ones to ``KeyServer``, a tiny HTTP endpoint of this
process on its own port. While the keyboard policy is not chosen the script lets every
key through; while it is, it keeps these keys from the 3D view (whose camera also
moves with W A S D and the arrows; the mouse still moves it). The browser reports
every 0.2 s; keys not heard about for ``TIMEOUT_S`` count as released, so a closed or
unfocused tab stops the instruments. Keys are matched by their place on the keyboard
(``event.code``), by their name when a browser leaves that empty.
"""

from __future__ import annotations

import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import FrozenSet, Iterable, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .i18n import tr
from .policies import _Base

LABEL = "键盘手动"
TIMEOUT_S = 0.6
# per instrument, in action order (push, rotate): (push +, push -, rotate +, rotate -)
KEYS = (("ArrowUp", "ArrowDown", "ArrowRight", "ArrowLeft"),
        ("KeyW", "KeyS", "KeyD", "KeyA"))


class KeyState:
    """The keys the browser says are held, and how fast they drive (share of the top speed)."""

    def __init__(self, timeout_s: float = TIMEOUT_S) -> None:
        self._lock = threading.Lock()
        self._held: FrozenSet[str] = frozenset()
        self._stamp = -np.inf
        self.timeout_s = timeout_s
        self.speed = 0.5

    def set(self, codes: Iterable[str]) -> None:
        with self._lock:
            self._held = frozenset(c for c in codes if c)
            self._stamp = time.monotonic()

    def held(self) -> FrozenSet[str]:
        with self._lock:
            return self._held if time.monotonic() - self._stamp < self.timeout_s else frozenset()


def keyboard_action(held: FrozenSet[str], action_dim: int, speed: float) -> np.ndarray:
    """Normalised action from the held keys: instrument k drives dimensions 2k (push), 2k+1 (rotate)."""
    action = np.zeros(action_dim, dtype=np.float32)
    for k, (push, pull, turn, back) in enumerate(KEYS[: action_dim // 2]):
        action[2 * k] = speed * ((push in held) - (pull in held))
        action[2 * k + 1] = speed * ((turn in held) - (back in held))
    return np.clip(action, -1.0, 1.0)


KEY_STATE = KeyState()          # the workbench's keys: KeyServer writes, KeyboardPolicy reads


class KeyboardPolicy(_Base):
    label = LABEL

    def __init__(self, action_dim: int, keys: KeyState) -> None:
        self.action_dim, self.keys = int(action_dim), keys

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        return keyboard_action(self.keys.held(), self.action_dim, float(self.keys.speed))


class KeyServer:
    """``GET /keys?s=ArrowUp,KeyW`` sets the held keys; runs in a daemon thread."""

    def __init__(self, keys: KeyState, host: str = "0.0.0.0", port: int = 0) -> None:
        state = keys

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server's name)
                url = urlparse(self.path)
                if url.path == "/keys":
                    state.set(parse_qs(url.query).get("s", [""])[0].split(","))
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

            def log_message(self, *args):  # quiet
                pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


_SCRIPT = """(function(){
var W=window; W.__wbKeyCfg=%(cfg)s;
if(W.__wbKeys){W.__wbKeys.refresh();return;}
var held=new Set(), ours=new Set(%(keys)s);
function send(){var c=W.__wbKeyCfg;
  fetch(location.protocol+'//'+location.hostname+':'+c.port+'/keys?s='+encodeURIComponent(Array.from(held).join(',')),
        {mode:'no-cors',cache:'no-store'}).catch(function(){}); show();}
function show(){var el=document.getElementById('wb-keys-held');
  if(el) el.textContent=held.size?Array.from(held).join(' '):'—';}
function code(e){if(e.code) return e.code;
  var k=e.key||''; return k.length===1?'Key'+k.toUpperCase():k;}
function on(e){var c=W.__wbKeyCfg, k=code(e);
  if(!c.active||!ours.has(k)) return;
  if(!c.two&&k.indexOf('Key')===0) return;
  e.preventDefault(); e.stopImmediatePropagation();
  var had=held.has(k);
  if(e.type==='keydown') held.add(k); else held.delete(k);
  if(had!==held.has(k)) send();}
W.addEventListener('keydown',on,true); W.addEventListener('keyup',on,true);
W.addEventListener('blur',function(){held.clear(); if(W.__wbKeyCfg.active) send();});
setInterval(function(){if(W.__wbKeyCfg.active) send();},200);
W.__wbKeys={refresh:function(){if(!W.__wbKeyCfg.active){held.clear();} send();}};
show();})();"""


def capture_html(port: Optional[int], active: bool, two: bool, lang: str = "zh") -> str:
    """The key legend and the script that reports the held keys (inactive: lets keys through)."""
    if port is None:
        return ""
    cfg = json.dumps({"port": port, "active": bool(active), "two": bool(two)})
    keys = json.dumps([k for group in KEYS for k in group])     # one instrument: W A S D pass (``two``)
    script = html.escape(_SCRIPT % {"cfg": cfg, "keys": keys}, quote=True)
    trigger = f'<img alt="" style="display:none" src="data:image/gif;base64,invalid" onerror="{script}">'
    if not active:
        return f'<div style="display:none">{trigger}</div>'
    legend = tr("keys_legend_two" if two else "keys_legend_one", lang)
    return (f'<div style="font-size:12px;line-height:1.5">{legend}<br>'
            f'<span style="opacity:.65">{tr("keys_held", lang)}</span> '
            f'<code id="wb-keys-held">—</code>{trigger}</div>')
