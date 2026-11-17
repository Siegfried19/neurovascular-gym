#!/usr/bin/env bash
# Install the training environment on a fresh Linux machine (Ubuntu 22.04 or newer, x86_64). Runs ON that machine;
# scripts/remote/provision.sh (on the laptop) copies the files over and then calls this. Safe to run again:
# every step that is already done is skipped.
#
#   bash scripts/remote/setup_machine.sh
#
# What comes from where:
#   official sources  Miniforge (conda-forge), Python, every pip package at the version in requirements.lock.txt,
#                     the SOFA v25.12.00 Linux binary for Python 3.12 (GitHub release, sha256 checked)
#   copied over       this repository, stEVE/, our BeamAdapter build (~/opt/BeamAdapter-v25.12-dev), demonstrations
# Only the new environment (guidewire_rl + SOFA 25.12 + our BeamAdapter); the old contact_aware_tdmpc2 one is not installed.
# Result: `conda run --no-capture-output -n guidewire_rl python ...` works as on our own machines.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPT="${OPT:-$HOME/opt}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME=guidewire_rl
PYTHON_VERSION=3.12.15                                  # the laptop's; falls back to the newest 3.12 if gone
SOFA="$OPT/SOFA_v25.12.00_Linux"
SOFA_URL=https://github.com/sofa-framework/sofa/releases/download/v25.12.00/SOFA_v25.12.00_Linux-Python_3.12.zip
SOFA_SHA256=76be4a5d89656a7e8c89bc45a30665aab2a52ba33e77e2a59280b735a9288574
BEAM="$OPT/BeamAdapter-v25.12-dev"
# System libraries the SOFA binary and the vtk / opencv / pygame wheels load (from ldd over SOFA's lib/ on the laptop);
# build-essential (gcc) for training.compile_update: torch.compile builds its kernel launchers with the C compiler.
APT_PACKAGES=(ca-certificates curl unzip rsync git build-essential libgl1 libglx0 libglvnd0 libopengl0 libegl1 libglib2.0-0
              libfontconfig1 libfreetype6 libxext6 libxrender1 libsm6 libgomp1 libbrotli1 liblz4-1 libzstd1
              libgcrypt20 libsystemd0 libwayland-client0 libwayland-cursor0)

T0=$SECONDS; LAST=$SECONDS
step() { echo; echo "=== $1"; }
took() { echo "--- $1: $((SECONDS - LAST)) s"; TIMES+=("$1: $((SECONDS - LAST)) s"); LAST=$SECONDS; }
TIMES=()

[ -d "$REPO/stEVE/eve" ] || { echo "no stEVE/ in $REPO (copy it over first: scripts/remote/provision.sh)"; exit 1; }
[ -f "$BEAM/BeamAdapter/lib/libBeamAdapter.so" ] || { echo "no BeamAdapter build at $BEAM (copy it over first)"; exit 1; }

step "system packages"
if command -v apt-get >/dev/null; then
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  if ! dpkg -s "${APT_PACKAGES[@]}" >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}"
  fi
else
  echo "no apt-get: install the equivalents of: ${APT_PACKAGES[*]}"
fi
took "system packages"

step "Miniforge"
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_ROOT" >/dev/null
  rm -f /tmp/miniforge.sh
fi
CONDA="$CONDA_ROOT/bin/conda"
# `conda` on PATH for login shells and for non-interactive ssh commands
grep -qs "$CONDA_ROOT/condabin" "$HOME/.bashrc" || sed -i "1i export PATH=\"$CONDA_ROOT/condabin:\$PATH\"" "$HOME/.bashrc" 2>/dev/null \
  || echo "export PATH=\"$CONDA_ROOT/condabin:\$PATH\"" >> "$HOME/.bashrc"
[ -w /usr/local/bin ] && ln -sf "$CONDA_ROOT/condabin/conda" /usr/local/bin/conda || true
took "Miniforge"

step "conda env $ENV_NAME + pip packages at the pinned versions"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA" create -y -q -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip \
    || "$CONDA" create -y -q -n "$ENV_NAME" -c conda-forge "python=3.12" pip
fi
# PYTHONNOUSERSITE: packages in ~/.local/lib/python3.12 must not shadow the env's (docs: todo_list plan, 2026-11-15)
export PYTHONNOUSERSITE=1
"$ENV_DIR/bin/python" -m pip install -q --no-deps -r "$REPO/requirements.lock.txt"
took "conda env and pip packages"

step "SOFA v25.12.00 (official binary)"
if [ ! -f "$SOFA/lib/libSofa.Core.so.25.12.00" ]; then
  mkdir -p "$OPT/.sofa_download"
  curl -fL --retry 3 -o "$OPT/.sofa_download/sofa.zip" "$SOFA_URL"
  echo "$SOFA_SHA256  $OPT/.sofa_download/sofa.zip" | sha256sum -c -
  unzip -q "$OPT/.sofa_download/sofa.zip" -d "$OPT/.sofa_download"
  mv "$OPT/.sofa_download/SOFA_v25.12.00_Linux" "$SOFA"
  rm -rf "$OPT/.sofa_download"
fi
took "SOFA"

step "editable installs (stEVE, this repository) and SOFA's variables in the env"
"$ENV_DIR/bin/python" -m pip install -q --no-deps -e "$REPO/stEVE" -e "$REPO"
# Order matters: our BeamAdapter first in SOFA_PLUGIN_PATH; the env's own lib last in LD_LIBRARY_PATH
# (SofaPython3 looks for libpython3.12.so.1.0 there).
"$CONDA" env config vars set -n "$ENV_NAME" \
  SOFA_ROOT="$SOFA" \
  SOFA_PLUGIN_PATH="$BEAM:$SOFA/plugins" \
  PYTHONPATH="$SOFA/plugins/SofaPython3/lib/python3/site-packages" \
  LD_LIBRARY_PATH="$SOFA/lib:$SOFA/plugins/SofaPython3/lib:$ENV_DIR/lib" \
  PYTHONNOUSERSITE=1 >/dev/null
took "editable installs and variables"

step "check"
cd "$REPO"
"$CONDA" run --no-capture-output -n "$ENV_NAME" python - <<'EOF'
import os, sys
import numpy, torch, gymnasium, eve, guidewire_rl
import Sofa, SofaRuntime
SofaRuntime.importPlugin("BeamAdapter")
beam = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libBeamAdapter" in line})
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("eve from", os.path.dirname(eve.__file__))
print("BeamAdapter loaded from", beam)
assert beam and all("BeamAdapter-v25.12-dev" in path for path in beam), "SOFA loaded its own BeamAdapter, not ours"
assert numpy.__version__ == "2.5.4" and torch.__version__.startswith("2.7.1"), "versions differ from the lock file"
EOF
took "check"

echo
echo "=== machine"
echo "cpu: $(nproc) threads, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "memory: $(free -g | awk '/^Mem/{print $2" GB total, "$7" GB available"}')"
echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $2" total, "$4" free"}')   /tmp: $(df -h /tmp | awk 'NR==2{print $4" free"}')   repo disk: $(df -h "$REPO" | awk 'NR==2{print $4" free"}')"
echo "glibc: $(ldd --version | head -1 | awk '{print $NF}')"
echo
echo "=== time"
printf '%s\n' "${TIMES[@]}"
echo "setup_machine.sh total: $((SECONDS - T0)) s"
