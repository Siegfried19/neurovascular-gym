# CLAUDE.md

这个文件给在本仓库里工作的 Claude Code / Codex 看。规则很短，细节都在 `README.md`。

1. **动手前把 `README.md` 从头读完。** 它写了这是什么、怎么装、怎么跑、每个文件夹管什么、改东西去哪、哪些文件不能手改、现在做到哪了。
2. **跑任何 Python 一律 `conda run --no-capture-output -n guidewire_rl python …`。** 直接调环境里的 `python` 拿不到 SOFA 的环境变量，`import Sofa` 会失败。
3. **不要手改 README「不能手改的文件」一节列的东西**（搬来的代码、带哈希锁的数据、BeamAdapter 官方补丁）。要换数据就重新生成并更新哈希。
4. **改完代码跑直接覆盖它的测试**（README「测试」一节有文件 → 测试的对照表）；提交前跑全量 `pytest -q`（约 20–30 分钟，含真实 SOFA 场景）。
5. 训练和评估的产物写到 `runs/`，不进仓库。一次性的图、探路脚本放在会话临时目录，也不进仓库。
6. 代码注释和配置注释用英文；`docs/` 和各文件夹的 `README.md` 是中文。回复用用户用的语言。
