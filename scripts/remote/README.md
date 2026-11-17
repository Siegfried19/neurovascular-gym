# scripts/remote：把一台新机器装成训练机

只装新环境这一套（`guidewire_rl` + SOFA 25.12 + 修过的 BeamAdapter）。旧环境不带。

| 文件 | 在哪跑 | 干什么 |
| --- | --- | --- |
| `provision.sh <主机> [端口]` | 笔记本 | 一条命令装好：把我们的文件传过去，再在那台机器上跑 `setup_machine.sh`；最后报传文件和总共花的时间 |
| `setup_machine.sh` | 新机器 | 装系统库、Miniforge、环境、SOFA，做两个 editable 安装，把 SOFA 的变量和 `PYTHONNOUSERSITE=1` 写进环境，自检，报每一步花的时间。可以重复跑，做过的步骤会跳过 |
| `start_run.sh <训练名> [train.py 的参数]` | 新机器 | 开训或续跑（后面加 `--resume <存档>`）：20 个仿真、wandb 离线、脱离 SSH 会话；`~/tools/exam_straggler_guard.py` 在的话一起挂上（考试被一两局算不动的仿真拖住时收尾） |
| `pull_run.sh <主机> <端口> <训练名>` | 笔记本 | 把一轮的记录、存档、离线 wandb 拉回笔记本的 `runs/<训练名>/`；只往笔记本拷，两边都不删。`LOGS_ONLY=1` 时不拉存档 |
| 根目录 `requirements.lock.txt` | — | `guidewire_rl` 里每个 pip 包的确切版本；新机器照它从官方源装 |

## 东西从哪来

- **官方源下载**：Miniforge 和 Python（conda-forge）、清单里的全部 pip 包（PyPI 和 PyTorch 的 cu128 源）、SOFA v25.12.00 的 Python 3.12 Linux 包（GitHub 发布页，脚本核对 sha256）。
- **从笔记本传**：git 追踪的和会追踪的文件（工作副本，没提交的改动、还没 add 的新文件也带上；`.gitignore` 排掉的不传）、`stEVE/`（不带 `.git`）、`~/opt/BeamAdapter-v25.12-dev`、示范文件夹（默认 `runs/demos_round2f`，用环境变量 `DEMOS` 换）。
- 不传任何钥匙（GitHub、wandb、Vast）。训练在那边用 `--set logging.wandb.mode=offline`，记录拉回笔记本再上传。

装好以后目录和我们自己的机器一样（相对家目录）：`~/Projects/contact-aware-tdmpc2`、`~/opt/SOFA_v25.12.00_Linux`、`~/opt/BeamAdapter-v25.12-dev`；命令也一样，走 `conda run --no-capture-output -n guidewire_rl …`。

## 要求

Ubuntu 22.04 或更新、x86_64（修过的 BeamAdapter 是在 glibc 2.35 上编的）；能用 root 或 sudo 装系统库；NVIDIA 显卡驱动由机器或容器提供（torch 自带 CUDA 库）。

## 环境里的包变了以后，重做清单

```bash
{ sed -n '/^#/p;/^--extra/p' requirements.lock.txt; \
  conda run -n guidewire_rl pip list --format=freeze | grep -vE '^(eve|guidewire-rl|pip)=='; } > /tmp/lock && mv /tmp/lock requirements.lock.txt
```

`setup_machine.sh` 自检里核对的 numpy、torch 版本号，和 SOFA 包的地址与校验码，写在脚本开头。
