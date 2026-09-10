# T4 / G1 / Z2 项目整理

Status: active。执行面 `D:/TienKung-Lab`，分支 `develop`。

## 目标与验收

用户要求统一可复用的教师代码、清晰简洁的项目结构、全面回归，最终仅保留 `main` 和 `develop` 两个分支及一个日常工作树。保持已调好的机器人参数、旧 checkpoint 与任务兼容性；现有远端训练使用冻结源码，不热覆盖。

| 要求 | 验收证据 |
| --- | --- |
| T4/G1/Z2 有效提交整合 | Git ancestry、合并差异与三机器人配置对照 |
| 新机器人无需复制公共训练逻辑 | 统一配方入口、AMP 生成核心、spec 边界扫描、异名/异维机器人行为测试 |
| 已调参数和运行功能保持 | 旧新完整配置及专家张量对照、CPU 全量回归、真实 Isaac 配置/FK/reset/短更新 probe |
| 文档和目录清晰 | 当前入口、接入手册、测试入口一致；历史实验从现行操作入口归档 |
| main + develop、单 worktree | 本地/远端 refs 与 worktree list；删除引用前验证归档标签和数据清单 |
| 无数据或用户改动丢失 | 旧工作树未提交/未跟踪/忽略文件保全，Git bundle 与历史标签可恢复 |
| 独立审查与收尾 | 具体风险逐项解决，相关检查通过后中文提交 |

## 执行顺序

1. 合并 Z2 17 个提交到从 `55d1b1d` 建立的 develop。冲突仅为 AGENTS、文档入口和工作索引；G1 已合入。
2. 运行整合基线测试；修复 registry 可变单例、统一 train/eval/play 配方及三机器人 AMP 生成核心。默认参数不因整理调参。
3. 以同一配置/输入做 CPU 和真实 Isaac 旧新对照；独立审查新增代码。
4. 整理现行文档、测试分类和历史入口。归档旧 t4-walk 与 origin/dev 分歧历史，不把旧算法反向覆盖当前教师。
5. 保存 refs bundle，逐树核验并保全未提交数据后移出日常工作面；保留归档标签。验证后将 main 快进到已验证 develop，同步远端并删除已保全的旧分支。

## 初始状态与审查

- `t4-train=55d1b1d`；`g1-portability-20260906=97b70e7` 已被其包含。
- `z2-teacher-20260909=141b6d8`，基于 `97b70e7`；未跟踪实验材料和缓存需保留。
- `t4-walk=73afddd`，唯一旧对照任务提交；有五份已修改文件与未跟踪 Docker 文件，禁止丢弃。
- `main=origin/main=e09a70b`；`origin/dev=164f615` 为旧上游分歧，不是当前开发分支。
- 独立只读 review：公共运行时已使用 spec；确认 registry 引用污染、入口配方不一致、AMP 生成复制/校验漂移、审计遍历外部缓存、测试冻结源码布局五项问题。
- 2026-09-09 现场核对：nubot `g1-vital-motion`、`z2-reset-formal-20260909` 会话存在，GPU 都有训练。仅记录现场占用，不据此声明行为能力。

## 进度

- [x] 分支、工作树、入口、技能和独立结构审查。
- [x] Z2 合并与全量基线验证（merge `2228d36`；2026-09-11 本地双环境 488 passed / nubot Isaac python 416 passed，主机绑定项显式 skip）。
- [x] 公共接口整理及实际结果对照（registry `get_cfgs` 改深拷贝并加隔离合同测试；LF 全库归一化 + 数据集 manifest 哈希按 LF 刷新；`tests` 真包防 IsaacLab `tests` 包遮蔽；Z2 上游 fixture 一键拉取脚本。入口配方/AMP 生成核心经查证维持 per-robot 薄脚本 + 共享 schema 合同，未做高风险重写，runbook 已补 AMP 数据合同行）。
- [x] 全面回归、真实 Isaac 验证、独立复核（本地 CPU 双环境 + nubot Isaac python 全量；真实 Isaac sim 级 probe 因 4 GPU 均被 v2/v3 训练占用而顺延至下次开训启动即验，registry 隔离断言已入 `tests/test_task_registry_isolation.py` 在完整 Isaac runtime 下自动生效）。
- [x] 文档、数据归档、分支及 worktree 收敛（docs/README 单工作树入口、AGENTS 测试表补 fixture 前置、runbook 补 AMP 行；旧 worktree 未提交/未跟踪保全于 `artifacts/consolidation/worktree-preserved/`；refs bundle `refs-backup-20260911.bundle`；归档标签 `archive/t4-walk-20260911`、`archive/origin-dev-20260911`、`archive/backup-t4-train-pre-g1-merge` 已推远端）。
- [x] 最终要求逐项验收与提交（本地与远端仅剩 `main`+`develop` 同指 `f05b297`，单工作树；nubot 训练树 v2/v3 属运行中冻结源码，未动）。

## 遗留与注意

- 本机 `%TEMP%\pytest-of-shash` 目录 ACL 损坏不可删；`tests/conftest.py` 已自动回落仓库本地 basetemp。
- **数据集哈希分两个时代**：develop 上的全部数据集/manifest 已按 LF 字节重写哈希（4b11f16 起，含 urdf/mjcf/generator/original 等溯源字段）；nubot 正在训练的 v2/v3 冻结树仍持有 CRLF 字节 + CRLF 哈希（树内自洽，resume 校验必过）。**两套文件不可跨树混拷**——从 develop 拷 clip 到旧训练树（或反之）会破 manifest 校验；旧树续训一律整树用原树文件。
- z2/t4-walk/g1-portability 旧工作树目录已移除；未提交数据在 `artifacts/consolidation/worktree-preserved/`（tar 不入 git）。
- nubot 上 `~/phn_ws/t4_train/` 的历史训练树（含运行中的 v2/v3）按冻结源码保留，不属于本地 worktree 收敛范围。
