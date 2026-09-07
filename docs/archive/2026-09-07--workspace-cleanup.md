# 2026-09-07 工作区整理

本次整理文档货架与历史临时操作文件，不改变训练状态或能力结论。

- 已归档 `done` 的 [表示先行学生计划](plans/2026-09-01--t4-s12-repr-first-distill-plan.md) 和 `superseded` 的 [旧蒸馏与 FT 计划](plans/2026-08-29--t4-s12-student-distill-ft-final-plan.md)。阶段执行段落作为历史记录保留。
- 24 个 S12 临时脚本、同步压缩包和文件清单，共 26 个文件（254708 字节），集中到本机 `artifacts/work/cleanup-20260907/s12-oneoff/`。未删除，移动前后 SHA-256 一致。
- 本机 `artifacts/work/cleanup-20260907/manifest.json` 保存原路径、归档路径、大小和校验值。复用前核查旧运行参数；按清单恢复原路径可保留脚本间依赖。
- 保留训练代码改动、G1 诊断及部署脚本、checkpoint、evaluator JSON、lineage、日志与回放。根目录 `an.txt` 用途不明，保留。没有删除环境、缓存或实验数据。
- 索引及规格的引用已在工作区修正；它们混有先前未提交修改，本次提交不纳入这些文件。

当前入口仍为 [文档索引](../README.md) 与 [工作面索引](../../.harness/work_index.md)。整理不代表 G1 能力验收或 T4 plant 重训完成。
