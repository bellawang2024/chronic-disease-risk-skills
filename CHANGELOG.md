# 更新日志 / Changelog

本项目所有重要变更都记录在此文件。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [1.0.0] — 2026-09-18

### 新增 — 慢病风险预测 Skills 技能包

慢病方向的首个正式版本，交付 **7 个 Agent Skills 技能包**，全部构建在公开可获取的人群调查数据之上。

| 技能包 | 疾病 | 数据来源 | 样本量 | 阳性率 | 特征数 | CV AUC | CV 平衡准确率 | 决策阈值 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `chd_prediction` | 冠心病 / 心肌梗死 | BRFSS 2017 | 445,872 | 8.83% | 15 | 0.8460 | 0.7684 | 0.08 |
| `copd_prediction` | 慢性阻塞性肺疾病 (COPD) | BRFSS 2017 | 447,718 | 8.39% | 14 | 0.8415 | 0.7641 | 0.09 |
| `stroke_prediction` | 脑卒中 | BRFSS 2017 | 448,666 | 4.22% | 15 | 0.8149 | 0.7436 | 0.04 |
| `hypertension_prediction` | 高血压 | NHANES 2017–2018 | 5,250 | 54.42% | 14 | 0.8234 | — | 0.5 |
| `diabetes_prediction_v3` | 糖尿病 (v3) | NHIS 2016–2025 | 291,945 | 10.88% | 7 | 0.8124 | 0.7419 | 0.105 |
| `hyperlipidemia_prediction` | 高脂血症 | NHIS 2016–2025 | 291,295 | 31.56% | 7 | 0.7956 | 0.7251 | 0.305 |
| `diabetes_prediction` | 糖尿病 (v2 · Pima) | Pima Indians | 768 | 34.90% | 8 | 0.8379 | — | 0.5 |

每个技能包均包含：

- `SKILL.md` — 技能说明（frontmatter + 完整方法论）
- `scripts/fetch_data.py` — 按 SHA256 校验下载原始数据
- `scripts/train_model.py` — 完整可复现训练流程
- `scripts/predict.py` / `batch_predict.py` — 推理脚本
- `models/` — 已训练模型（推理开箱即用）
- `results/` — 交叉验证与留出集完整指标
- `data/DATA_PROVENANCE.md` — 数据来源、许可证、校验值

### 新增 — 工具与文档

- `tools/build_skill_index.py` — 从模型产物自动生成 `skills/README.md` 索引，支持 `--check` 校验表格与模型是否一致
- `tools/build_changelog.py` — 生成本文件
- docs/METHODOLOGY.md（方法论）
- docs/DATA_SOURCES.md（数据来源与选型调研）
- docs/PERFORMANCE.md（全部技能包性能指标）
- `CITATION.cff`、`LICENSE`（MIT）、`NOTICE.md`（数据说明与医疗免责）、`CONTRIBUTING.md`

### 变更

- 交付物定位由「模型集」调整为「**Agent Skills 技能包**」，仓库更名为 `chronic-disease-risk-skills`
- 中英文 README、`CITATION.cff`、各文档标题同步更新，明确最终交付物为技能包、且全部基于公开数据

### 说明

- 原始数据集 CSV **不纳入 git**（本仓库合计 约 69 MB）。改由各技能包的 `scripts/fetch_data.py` 按 SHA256 校验下载 —— 校验和比直接提交文件更能证明「拿到的确实是同一份数据」
- `docs/PERFORMANCE.md` 与 `skills/README.md` 中的全部指标均由模型产物自动生成，无手工转录
- 模型中的特征名均为真实变量名（自定义 `NamedXGB` 子类写入 `feature_names`），不会出现 `f0, f1, ...`

---

## 免责声明

**本项目仅用于研究与教学，不构成医疗建议，不得用于临床诊断。**
所有模型反映的都是流行病学数据上的统计关联，个体风险判断请咨询执业医师。
