# 贡献指南 / Contributing

[简体中文](#简体中文) | [English](#english)

---

## 简体中文

感谢你有兴趣贡献！本项目欢迎新的疾病预测模型，但**有明确的质量门槛**。

### 提交新模型前，请确认以下四条

**1. 数据必须权威、公开、可追溯**

- 优先使用政府/学术机构的公开数据（CDC、NCHS、NCI、国家队列等）
- **必须提供逐文件的校验值（SHA256/MD5）**，并用 `fetch_data.py` 自动校验
- 如果数据需要申请/DTA，请在文档中写清申请流程与限制（这本身是有价值的信息）

**2. 必须遵循四条设计原则**

| 原则 | 要求 |
|------|------|
| 不使用诊断性指标作特征 | 如果该指标本身就是该病的诊断依据（如用 HbA1c 预测糖尿病），必须排除。请在文档中说明排除了什么、为什么 |
| 不使用人群划分属性 | 性别、种族等在**数据构建阶段**就不应提取。如确有例外（如队列内为常量），必须说明理由 |
| 不使用体现代理变量 | 教育、收入、体检频率等是"谁更容易被诊断"的代理，不是病因 |
| 稀有结局不用普通准确率 | 阳性率 <10% 时必须报告 AUC / PR-AUC / 平衡准确率 |

**3. 必须做特征筛选，并公开实测依据**

不能只写"根据文献纳入了 N 个危险因素"。请用**特征阶梯**：
从最小特征集开始分层加入，用相同切分 + 多个随机种子评估，**只保留有实质增益的特征**，并公开每层的实测数值。

**4. 必须公开验收线的依据**

本项目**不使用统一的 AUC 门槛**。如果你的模型低于 0.80，这**不一定**是拒绝理由——
但你必须说明：为什么这条线是合理的，以及数据本身的上限在哪里。

> 如果模型确实受限于数据（如关键危险因素未被采集），**我们欢迎把它作为"诚实的阴性结果"提交**，
> 但必须在 README 中显著标注其局限，不能与高性能模型并列展示而不加区分。

### 必须附带的文件

```
<disease>_prediction/
├── SKILL.md                 # 技能说明，含完整评估与解读警告
├── README.md
├── data/DATA_PROVENANCE.md  # 必需：数据来源、SHA256、变量映射、标签规则、特征取舍、局限
├── models/                  # 已训练模型（json + ubj + imputer + metadata）
├── results/                 # 全部评估指标
└── scripts/                 # fetch_data / train_model / predict / batch_predict / feature_importance
```

### 提交前自检清单

- [ ] `python scripts/fetch_data.py` 能从官方地址重建数据，SHA256 一致
- [ ] `python scripts/train_model.py` 能复现模型工件（逐字节一致）
- [ ] 删除 `data/` 和 `results/` 后，`predict.py` 仍能正常推理
- [ ] 阈值仅在训练集内选择，未接触测试集
- [ ] README/SKILL 中的每个数字都能在 `results/*.json` 中找到出处
- [ ] 明确写出「本模型不能替代临床诊断」

### 流程

1. Fork → 新建分支 `add-<disease>-model`
2. 按上述结构提交
3. 在 PR 描述中说明：数据来源、样本量、阳性数、CV AUC、验收线及其依据
4. 维护者会检查是否满足四条原则与自检清单

---

## English

Thanks for your interest in contributing! New disease models are welcome, but there is a **clear quality bar**.

### Before submitting a new model

**1. Data must be authoritative, public and traceable**

- Prefer public data from government or academic institutions (CDC, NCHS, NCI, national cohorts)
- **Per-file checksums (SHA256/MD5) are required**, verified automatically in `fetch_data.py`
- If the data requires an application or DTA, document the process and its restrictions — this is valuable information in itself

**2. The four design principles are mandatory**

| Principle | Requirement |
|-----------|-------------|
| No diagnostic criteria as features | If the measurement *is* the diagnostic criterion (e.g. HbA1c for diabetes), exclude it. Document what was excluded and why |
| No demographic partitioning attributes | Sex, race, etc. must not be extracted **at the data-construction stage**. Document any exception (e.g. a constant within the cohort) with justification |
| No healthcare-access proxies | Education, income and screening frequency predict *who gets diagnosed*, not *who gets sick* |
| No plain accuracy for rare outcomes | When prevalence <10%, report AUC / PR-AUC / balanced accuracy |

**3. Feature selection must be evidence-based and published**

Do not write "N risk factors were included based on the literature". Use a **feature ladder**: start minimal, add in layers, evaluate on the same split across multiple random seeds, **keep only features with a material gain**, and publish the numbers for every layer.

**4. The acceptance threshold must be justified**

This project **does not use a single AUC cut-off**. Falling below 0.80 is **not** automatically grounds for rejection — but you must explain why your threshold is reasonable and where the data-imposed ceiling lies.

> If the model is genuinely data-limited (e.g. key risk factors are not collected), **we welcome it as an honest negative result** — but its limitations must be prominently labelled and it must not be presented alongside high-performing models without distinction.

### Required files

```
<disease>_prediction/
├── SKILL.md                 # Skill docs with full evaluation and interpretation warnings
├── README.md
├── data/DATA_PROVENANCE.md  # Required: sources, SHA256, variable mapping, label rules, feature rationale, limitations
├── models/                  # Trained artefacts (json + ubj + imputer + metadata)
├── results/                 # All evaluation metrics
└── scripts/                 # fetch_data / train_model / predict / batch_predict / feature_importance
```

### Pre-submission checklist

- [ ] `python scripts/fetch_data.py` rebuilds the data from official endpoints with matching SHA256
- [ ] `python scripts/train_model.py` reproduces the model artefacts byte-for-byte
- [ ] `predict.py` still works after deleting `data/` and `results/`
- [ ] The threshold is selected on training data only
- [ ] Every number in README/SKILL is traceable to `results/*.json`
- [ ] The statement "this model cannot replace clinical diagnosis" is present

### Process

1. Fork → create a branch `add-<disease>-model`
2. Submit following the structure above
3. In the PR description, state: data source, N, case count, CV AUC, threshold and its rationale
4. Maintainers will check the four principles and the checklist
