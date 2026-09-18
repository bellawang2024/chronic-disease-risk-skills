---
name: hypertension_prediction
description: "高血压风险预测模型 - 基于真实权威全人群数据集 NHANES 2017-2018（美国 CDC/NCHS 全国代表性抽样，5,250 名成人 14 项常规指标）训练 XGBoost 二分类模型，留出测试集准确率 0.7562、AUC 0.8209，10×5 折交叉验证准确率 0.7597、AUC 0.8234。不使用性别、种族等任何人群划分属性。触发条件：用户提到高血压预测、血压风险评估、高血压筛查、心血管风险分析等。"
version: 1.1.0
license: 仅供学习研究使用（数据为美国 CDC 公开数据）
metadata:
  display_name: 高血压风险预测
  category: 技能开发
---

> **Important:** All `scripts/` paths are relative to this skill directory.
> Run with: `cd {this_skill_dir} && python scripts/...`
> Or use the `cwd` parameter of `execute_shell_command`.

# 高血压风险预测模型

## 概述

基于**真实、可追溯、全人群**的公开权威数据集 NHANES 2017-2018，
使用 XGBoost 训练高血压风险二分类模型。本 skill 内置**已训练好的推理模型**，
解压即可预测，无需重新训练。

| 项目 | 值 |
|------|-----|
| 任务 | 二分类（是否患高血压，ACC/AHA 2017 标准） |
| 算法 | XGBoost (`xgboost.XGBClassifier`) |
| 数据集 | NHANES 2017-2018（美国 CDC / NCHS） |
| **目标人群** | **美国非机构化平民人口，全国代表性抽样，不限性别/种族** |
| 样本数 | **5,250** 名成人（真实临床测量数据） |
| 特征数 | **14**（常规体检 + 问卷 + 血检） |
| 类别分布 | 阳性 2,857 / 阴性 2,393（患病率 54.4%） |
| 划分 | test_size=0.2, random_state=42, stratify=True → 4,200 训练 / 1,050 测试 |
| 决策阈值 | 0.5 |

**不使用性别、种族等任何人群划分属性**：本模型特征全部为年龄、体格测量、
生活方式与常规血检指标，可适用于任意性别与族裔个体。

## 数据权威性与可追溯性

| 环节 | 内容 |
|------|------|
| 调查项目 | NHANES（美国国家健康与营养调查） |
| 执行机构 | U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS) |
| 调查周期 | 2017–2018（cycle J） |
| 目标人群 | 美国非机构化平民人口（全人群，不限性别、种族、地区、收入） |
| 抽样方式 | 复杂多阶段概率抽样，全国代表性 |
| 数据许可 | 美国政府公开数据（Public Domain） |
| 官方页面 | https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017 |

**关键优势**：血压由经过培训的技术人员**实测**（非自我报告），
且抽样覆盖全人群。本建模队列中男性 2,549 人、女性 2,701 人。

**校验机制**：使用的 10 个 CDC 原始文件均记录 SHA256 并内置于 `scripts/fetch_data.py`，
下载时自动校验，任一文件不符即中止。派生数据集校验值：

| 文件 | SHA256 |
|------|--------|
| `data/hypertension_nhanes.csv` | `e08ec328a3cd6cec11b8beee90352753cc3db887857944c14a591252e1b2be5f` |

已实测：删除内置 CSV 后重新从 CDC 下载全部 10 个文件并重建，
产出文件与内置文件**逐字节一致**。详见 `data/DATA_PROVENANCE.md`。

## 依赖

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

可选（仅绘图 / 重建数据时需要）：`pip install matplotlib`（重建数据仅需 pandas + numpy）

## 功能

### 1. 预测个体风险（模型已内置，无需先训练）

只需 **3 个核心指标**即可预测，其余自动用训练中位数插补：

```bash
python scripts/predict.py --age 55 --bmi 27.5 --waist 95
```

提供更多指标可获得更准确的评估：

```bash
python scripts/predict.py --age 55 --bmi 27.5 --waist 95 \
    --hba1c 6.1 --cholesterol 210 --hdl 42 --creatinine 1.0 --uric-acid 6.8 \
    --smoker 1 --diabetes 0 --pulse 78 --height 172 --income-pir 2.5
```

参数说明：

| 参数 | 描述 | 单位 | 合理范围 | 是否必需 |
|------|------|------|----------|----------|
| `--age` | 年龄 | 岁 | 18–80 | **必需** |
| `--bmi` | BMI 体重指数 | kg/m² | 10–90 | **必需** |
| `--waist` | 腰围 | cm | 40–200 | **必需** |
| `--height` | 身高 | cm | 120–220 | 可选 |
| `--education` | 教育程度 | 1–5 | 1–5 | 可选 |
| `--income-pir` | 家庭收入贫困比 | 比值 | 0–5 | 可选 |
| `--smoker` | 是否吸烟（一生≥100支） | 1/0 | 0–1 | 可选 |
| `--diabetes` | 是否患糖尿病 | 1/0 | 0–1 | 可选 |
| `--pulse` | 静息脉搏 | 次/分 | 20–200 | 可选 |
| `--hba1c` | 糖化血红蛋白 | % | 3–20 | 可选 |
| `--cholesterol` | 总胆固醇 | mg/dL | 50–600 | 可选 |
| `--hdl` | 高密度脂蛋白 | mg/dL | 5–200 | 可选 |
| `--creatinine` | 血清肌酐 | mg/dL | 0.1–20 | 可选 |
| `--uric-acid` | 血清尿酸 | mg/dL | 0.5–20 | 可选 |

> 可选指标未提供时用训练中位数插补，输出中的 `imputed_features` 会明确列出
> 哪些指标被插补，保证结果透明。

### 2. 批量预测

```bash
python scripts/batch_predict.py --input examples/people.csv --output results.csv
```

输入 CSV 必需列 `age, bmi, waist_cm`，其余列可省略。示例：

```csv
age,bmi,waist_cm,hba1c,cholesterol,hdl,creatinine,uric_acid,smoker,diabetes,pulse,height_cm,education,income_pir
34,24.1,82,5.2,178,52,0.95,5.6,0,0,68,170,4,3.20
58,31.8,102,6.8,215,44,0.82,5.1,0,1,76,162,3,1.85
```

### 3. 训练 / 重新验证模型

```bash
python scripts/train_model.py
```

脚本会依次执行：留出测试集评估 → 10×5 折重复交叉验证 → 验收标准检查 →
用全量 5,250 条样本重训发布模型，并把全部参数与指标写入 `models/model_metadata.json`。

| 参数 | 说明 |
|------|------|
| `--data` | 外部训练 CSV |
| `--quick` | 减少交叉验证重复次数，加快运行 |
| `--include-sensitive` | 加入性别/种族特征（默认**不含**） |
| `--quiet` | 静默模式 |

未达验收标准时脚本以退出码 2 结束。

### 4. 特征重要性

```bash
python scripts/feature_importance.py
python scripts/feature_importance.py --plot results/feature_importance.png
```

### 5. 数据下载、校验与重建

```bash
python scripts/fetch_data.py            # 从 CDC 重新下载 + 校验 + 重建数据集
python scripts/fetch_data.py --check    # 仅校验内置 CSV
```

### 6. Python 代码调用

```python
import sys
sys.path.insert(0, "scripts")
from predict import predict_patient

result = predict_patient(age=55, bmi=27.5, waist_cm=95)
print(result["probability_pct"], result["label"], result["risk_level"])
# 56.12% 高血压 中等风险
```

## 训练参数（已记录，可复现）

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
)
```

超参数由**训练集内层 5 折 GridSearchCV 按 AUC 选出**（未使用测试集调参）。

预处理：`SimpleImputer(strategy="median")`。XGBoost 对特征尺度不敏感，
**无需标准化**。插补中位数：

| 特征 | age | bmi | waist_cm | height_cm | education | income_pir | smoker |
|------|-----|-----|----------|-----------|-----------|------------|--------|
| 中位数 | 51.0 | 28.4 | 98.9 | 166.0 | 4.0 | 2.11 | 0.0 |

| 特征 | diabetes | pulse | hba1c | cholesterol | hdl | creatinine | uric_acid |
|------|----------|-------|-------|-------------|-----|------------|-----------|
| 中位数 | 0.0 | 70.0 | 5.6 | 183.0 | 51.0 | 0.84 | 5.3 |

完整参数、指标、特征顺序、环境版本均记录在 `models/model_metadata.json`。

## 模型评估结果（实测）

### 留出测试集（1,050 样本，未参与训练与调参）

| 指标 | 数值 | 验收标准 | 结论 |
|------|------|----------|------|
| 准确率 | **0.7562** | ≥ 0.75 | 通过 |
| AUC | **0.8209** | ≥ 0.80 | 通过 |
| 精确率 | 0.7397 | — | — |
| 召回率 | 0.8511 | — | — |
| F1 分数 | 0.7915 | — | — |

混淆矩阵：TN=308, FP=171, FN=85, TP=486

### 10×5 折重复交叉验证（全量 5,250 样本，50 折）

| 指标 | 均值 ± 标准差 | 验收标准 | 结论 |
|------|---------------|----------|------|
| AUC | **0.8234 ± 0.0113** | ≥ 0.80 | 通过 |
| 准确率 | **0.7597 ± 0.0106** | ≥ 0.75 | 通过 |
| 精确率 | 0.7520 ± 0.0113 | — | — |
| 召回率 | 0.8338 ± 0.0147 | — | — |
| F1 | 0.7906 ± 0.0091 | — | — |

召回率 0.83 而精确率 0.75，偏向"宁可多筛不可漏筛"，适合筛查场景定位。

> **关于发布模型**：上表指标来自 4,200/1,050 划分与交叉验证；
> 随 skill 发布的模型用**全量 5,250 条**样本重训（生产惯例）。
> 因此以留出测试集与交叉验证指标为准，不要用训练集自检值评估泛化性能。

### 关于不使用性别/种族

| 特征方案 | 特征数 | CV AUC | CV 准确率 |
|----------|--------|--------|-----------|
| **默认（不含性别/种族）** | **14** | **0.8234** | **0.7597** |
| 含性别/种族（`--include-sensitive`） | 16 | 0.8301 | 0.7594 |

纳入性别与种族仅带来 +0.007 AUC，准确率基本持平。因此默认模型不使用任何
人群划分属性，适用人群更广、也避免了对特定群体的差异化输出。

## 特征重要性（实测，临床合理）

| 排序 | 特征 | 中文名 | 重要性 |
|------|------|--------|--------|
| 1 | age | 年龄 | 0.3129 |
| 2 | diabetes | 糖尿病 | 0.1010 |
| 3 | hba1c | 糖化血红蛋白 | 0.1001 |
| 4 | waist_cm | 腰围 | 0.0942 |
| 5 | uric_acid | 血清尿酸 | 0.0591 |
| 6 | bmi | BMI | 0.0476 |
| 7 | height_cm | 身高 | 0.0429 |
| 8 | pulse | 静息脉搏 | 0.0425 |
| 9 | creatinine | 血清肌酐 | 0.0398 |
| 10 | education | 教育程度 | 0.0376 |
| 11 | cholesterol | 总胆固醇 | 0.0325 |
| 12 | smoker | 吸烟 | 0.0312 |
| 13 | income_pir | 收入贫困比 | 0.0298 |
| 14 | hdl | HDL | 0.0287 |

年龄是首要风险因素，其次是糖代谢异常（糖尿病、HbA1c）与中心性肥胖（腰围），
与高血压流行病学认识一致，可作为模型合理性的旁证。

## 推理模型工件

| 文件 | 说明 |
|------|------|
| `models/hypertension_xgb_model.json` | XGBoost 原生模型（**推理首选**，跨版本可移植） |
| `models/hypertension_xgb_model.ubj` | XGBoost 二进制模型（等价备用） |
| `models/imputer.joblib` | 中位数插补器（备用） |
| `models/model_metadata.json` | 全部训练参数、指标、插补中位数、特征顺序、环境版本 |
| `results/holdout_metrics.json` | 留出测试集指标 |
| `results/cv_metrics.json` | 交叉验证指标 |
| `results/feature_importance.json` | 特征重要性 |
| `data/hypertension_nhanes.csv` | 内置训练数据（5,250 条） |
| `data/DATA_PROVENANCE.md` | 数据溯源、校验值、构建规则、使用限制 |

推理时**特征顺序**固定为 `model_metadata.json` 中
`preprocessing.feature_order` 的顺序（默认 14 项）。

推理只需 XGBoost + numpy：即使 `joblib` / `imputer.joblib` 缺失，
也能仅凭元数据中记录的插补中位数完成预测，结果完全一致
（已实测：手工插补与 `SimpleImputer` 在 5,250×14 = **73,500 个值上逐值完全一致**）。

## 输出结果示例

```json
{
  "probability": 0.5612,
  "probability_pct": "56.12%",
  "prediction": 1,
  "label": "高血压",
  "risk_level": "中等风险",
  "advice": "建议控制体重与腰围、减少钠盐摄入、增加运动，并定期测量血压",
  "decision_threshold": 0.5,
  "imputed_features": ["height_cm", "education", "income_pir"]
}
```

风险等级划分：

- 风险概率 < 30%：低风险
- 风险概率 30%–70%：中等风险
- 风险概率 > 70%：高风险

## 注意事项

1. **人群适用性**：数据来自**美国**非机构化平民人口调研（2017–2018），
   迁移到其他国家和地区人群时需谨慎并重新验证。
2. **仅成人**：队列只含 ≥18 岁成人，不适用于儿童青少年。
3. **标签口径**：高血压判定采用 ACC/AHA 2017 的 **130/80 mmHg** 标准
   （含正在服药者）。若采用 JNC7 的 140/90 标准，患病率与指标会不同。
4. **精度上限**：标签源自单次访视的实测血压，存在生理波动与"白大衣效应"，
   这是本任务 AUC 难以大幅提升的主要原因（文献中同类模型 AUC 多在 0.80–0.85）。
5. **未使用抽样权重**：训练未使用 NHANES 复杂抽样权重，
   模型输出**不可解释为全国患病率估计**，仅用于个体风险分层。
6. **临床决策**：本模型仅作为健康风险筛查的辅助参考与教学演示，
   **不能替代专业医生的诊断**。
