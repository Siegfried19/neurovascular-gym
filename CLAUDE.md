# CLAUDE.md

For Claude Code / Codex working in this repository. The rules are short; the details are in `README.md`.

1. **Read `README.md` to the end before changing anything.** It says what this is, how to install and run it, what every folder is for, where to make which kind of change, which files must not be edited by hand, and where the work stands.
2. **Run every Python command as `conda run --no-capture-output -n guidewire_rl python …`.** Calling the env's `python` by its path does not pick up SOFA's environment variables and `import Sofa` fails.
3. **Do not edit the files listed in the README under "Files that must not be edited by hand"** (copied code, hash-locked data, the official BeamAdapter patches). To replace data, regenerate it and update the hashes.
4. **After a code change, run the tests that cover it directly** (the README's "Tests" table maps files to tests); before committing, run the full `pytest -q` (about 20-30 minutes, includes real SOFA scenes).
5. Training and evaluation outputs go to `runs/` and are not committed. One-off figures and probe scripts belong in the session's scratch directory, not in the repository.
6. Code and config comments are in English; `docs/` and the per-folder `README.md` files are in Chinese. Reply in the language the user writes in.
