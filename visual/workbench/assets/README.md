# 工作台用的外部文件

| 文件 | 是什么 | 来源 | 授权 | sha256 |
| --- | --- | --- | --- | --- |
| `makehuman_base.obj` | MakeHuman 的基础人体网格（中性成年人，站立、手臂斜向下张开；单位 dm，x 向左、y 向上、z 向前）。工作台只用里面的 `body` 组（皮肤表面，13 380 个点、13 378 个四边形），再截掉手臂和腿；眼睛、牙齿、舌头、穿衣辅助网格和关节点都不画，关节点只拿来定位（`joint-neck`、`joint-head`、`joint-l-clavicle`、两边的 `shoulder` 和 `elbow`） | `makehumancommunity/makehuman` 仓库 `master` 分支 `makehuman/data/3dobjs/base.obj`，2026-11-06 原样下载 | CC0 1.0（文件头和仓库 `LICENSE.md`：随附资产 2020 年 9 月起放弃版权） | `8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c` |

怎么摆到 phantom 周围见 `parts/body.py` 开头。
