#!/usr/bin/env bash
# Build our BeamAdapter: upstream v25.12 (the commit SOFA 25.12 ships) + PATCHES (two upstream fixes, four ours),
# against the SOFA 25.12 binary in ~/opt, installed where the conda env `guidewire_rl` looks first
# (its SOFA_PLUGIN_PATH starts with PREFIX). Why: docs/simulator_changes.md, problems 1, 8, 17, 22 and 23.
#
#   bash simulator_modifications/beamadapter/build.sh
#   ASSERTIONS=1 bash simulator_modifications/beamadapter/build.sh   # bounds-checked std::vector (debugging)
#
# build/ next to this script holds the upstream checkout and the CMake build; it is not part of
# our source and can be deleted at any time (the next run clones it again).
# A new patch: edit build/src with the others applied and staged (git add -A), then git -C build/src diff > <name>.patch.
# Same compiler (system GCC 11.4) and Eigen (3.3.7) as the SOFA 25.12 Linux release.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES=("$HERE/official_irc_fix.patch"            # problem 1: push of exactly 0.01 mm crashes (PR 236)
         "$HERE/official_rod_straight_fix.patch"    # problem 17: straight section not first, rest shape off (PR 224)
         "$HERE/min_beam_length.patch"              # problem 22: ours, minBeamLength (default 0: unchanged)
         "$HERE/adaptive_mapping_refresh.patch"     # problem 8: ours, per-device collision points re-mapped
                                                    #   at the start of the step (only when listening)
         "$HERE/densest_sampling.patch"             # problem 23: ours, densestSampling (default off: unchanged)
         "$HERE/collision_start_abscissa.patch")    # problem 26: ours, AdaptiveBeamMapping startCurvAbs (default 0: unchanged):
                                                    #   no collision point on the part of a device sheathed by another
COMMIT=5d02b8322cc82fc8a04085862affb99b008ef175        # BeamAdapter branch v25.12, as shipped in SOFA 25.12
SRC="$HERE/build/src"
BUILD="$HERE/build/cmake"
SOFA_ROOT="${SOFA_ROOT_2512:-$HOME/opt/SOFA_v25.12.00_Linux}"
PREFIX="${PREFIX:-$HOME/opt/BeamAdapter-v25.12-dev}"
EIGEN="${EIGEN:-$HOME/opt/eigen-3.3.7}"
TOOLS="${TOOLS:-$HOME/anaconda3/envs/sofa-build}"      # cmake + ninja + boost headers

[ -f "$SOFA_ROOT/lib/cmake/Sofa.Config/Sofa.ConfigConfig.cmake" ] || { echo "no SOFA 25.12 at $SOFA_ROOT"; exit 1; }

# 1. upstream source at the pinned commit, plus the patches
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$HERE/build"
  git clone -q https://github.com/sofa-framework/BeamAdapter.git "$SRC"
  git -C "$SRC" checkout -q "$COMMIT"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "$SRC is not at $COMMIT; delete build/ and rerun"; exit 1; }
# Always from the pinned commit, patches in order: later patches change the context of earlier ones, so
# "is this one already applied" cannot be told patch by patch once several are in. Anything edited in
# build/src by hand is discarded here: save it as a patch and list it in PATCHES first.
git -C "$SRC" reset -q --hard "$COMMIT"
for PATCH in "${PATCHES[@]}"; do
  git -C "$SRC" apply "$PATCH" && echo "applied $(basename "$PATCH")"
done

# 2. Eigen 3.3.7 headers (header-only)
if [ ! -f "$EIGEN/Eigen/Core" ]; then
  mkdir -p "$(dirname "$EIGEN")"
  curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz | tar xz -C "$(dirname "$EIGEN")"
fi

# 3. configure, build, install
FLAGS=""
[ "${ASSERTIONS:-0}" = "1" ] && FLAGS="-D_GLIBCXX_ASSERTIONS"
mkdir -p "$BUILD"
"$TOOLS/bin/cmake" -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$TOOLS/bin/ninja" -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DCMAKE_PREFIX_PATH="$SOFA_ROOT/lib/cmake" -DSOFA_ROOT="$SOFA_ROOT" \
  -DEIGEN3_INCLUDE_DIR="$EIGEN" -DBoost_INCLUDE_DIR="$TOOLS/include" \
  -DBEAMADAPTER_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX/BeamAdapter" -Wno-dev > "$BUILD/configure.log" 2>&1 \
  || { tail -30 "$BUILD/configure.log"; exit 1; }
"$TOOLS/bin/ninja" -C "$BUILD" install > "$BUILD/ninja.log" 2>&1 || { grep -E "error|FAILED" "$BUILD/ninja.log" | head -30; exit 1; }

echo "BeamAdapter ${COMMIT:0:8} + $(for p in "${PATCHES[@]}"; do basename "$p"; done | paste -sd' ') installed to $PREFIX/BeamAdapter"
ls -l "$PREFIX/BeamAdapter/lib/libBeamAdapter.so"*
