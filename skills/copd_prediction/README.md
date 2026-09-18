# 慢性阻塞性肺疾病（COPD）风险预测 Skill（copd_prediction）

基于**真实权威全人群数据集** BRFSS 2017（美国 CDC 全美电话调查）
训练 XGBoost 二分类模型的 COPD 风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 关键指标

| 项目 | 值 |
|------|-----|
| 数据集 | BRFSS 2017（美国 CDC，50 州及属地成年居民抽样） |
| **目标人群** | **全人群，不限性别、种族**（性别/种族在数据构建阶段即未提取） |
| 原始 / 建模记录 | 450,016 → **447,718** |
| 特征 | 14 项（年龄、体格、生活方式、疾病史） |
| 标签 | BRFSS 核心变量 `CHCCOPD1`：曾被医生告知患有 COPD/肺气肿/慢性支气管炎 |
| 阳性事件 | **37,577**（患病率 8.39%） |
| 决策阈值 | **0.09**（训练集 CV 选定，非 0.5） |
| 留出测试集（89,544） | **AUC 0.8389**，**平衡准确率 0.7609**，敏感度 0.7513，特异度 0.7705 |
| 3×5 折交叉验证 | **AUC 0.8415 ± 0.0026**，**平衡准确率 0.7641 ± 0.0025** |
| 验收标准 | AUC ≥ 0.80、平衡准确率 ≥ 0.70 —— **全部达标** |

## ⚠️ 为什么不用"准确率"

本任务阳性率仅 8.39%，**多数类基线准确率就有 91.61%**。
本模型的普通准确率是 0.7689，**反而低于该基线** —— 这是不平衡分类中
准确率指标失效（accuracy paradox）的正常现象：把阈值调高、几乎不预测阳性，
准确率很容易"刷"到 91% 以上，但那样的模型毫无筛查价值。

因此验收采用 **AUC + 平衡准确率 + PR-AUC + 敏感度/特异度**。

## ⚠️ 确诊必须靠肺功能检查

COPD 的临床确诊必须依靠**肺功能检查**（吸入支气管舒张剂后 FEV1/FVC < 0.70）。
本模型仅基于问卷与常规指标，**不能用于确诊**，只能提示"是否值得去做肺功能检查"。

另外，COPD 在人群中**诊断率偏低**，标签是"曾被医生告知"，
因此**"阴性"不等于没有 COPD**，只表示未被告知过。

## 目录结构

```
copd_prediction/
├── SKILL.md                        # skill 定义与完整使用说明（含标准 frontmatter）
├── README.md                       # 本文件
├── requirements.txt
├── data/
│   ├── copd_brfss.csv              # 447,718 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md          # 溯源、校验值、编码核对、泄漏防控、限制
├── scripts/
│   ├── fetch_data.py               # 从 CDC 下载 + SHA256 校验 + 重建
│   ├── train_model.py              # 训练 + 验证 + 保存全部参数
│   ├── predict.py                  # 单个体推理
│   ├── batch_predict.py            # 批量推理
│   └── feature_importance.py       # 特征重要性
├── models/
│   ├── copd_xgb_model.json         # XGBoost 原生模型（推理首选）
│   ├── copd_xgb_model.ubj
│   ├── imputer.joblib
│   └── model_metadata.json         # 全部参数 / 指标 / 阈值 / 插补中位数
├── results/
│   ├── holdout_metrics.json
│   ├── cv_metrics.json
│   ├── feature_importance.json
│   └── batch_results.csv
└── examples/
    └── people.csv
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 个体预测（6 项主要指标）
python scripts/predict.py --age 62 --bmi 26.5 --smoker 1 --diabetes 0 \
    --hypertension 1 --high-chol 0

# 3. 批量预测
python scripts/batch_predict.py --input examples/people.csv --output results/batch_results.csv

# 4.（可选）重新训练并验证
python scripts/train_model.py

# 5.（可选）从 CDC 重新下载并校验数据（约 107MB，需 1-2 分钟）
python scripts/fetch_data.py
```

## 训练参数（已完整记录）

```python
XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
              subsample=0.9, colsample_bytree=0.9, min_child_weight=10,
              random_state=42, eval_metric="logloss", tree_method="hist")
# 预处理：SimpleImputer(strategy="median")，无需标准化
# 划分：train_test_split(test_size=0.2, random_state=42, stratify=y) → 358,174/89,544
# 阈值：训练集 5 折 CV 最大化平衡准确率 → 0.09
```

完整记录见 `models/model_metadata.json`。

## 数据质量旁证

COPD 率与吸烟、年龄的关系完全符合病因学：

| 吸烟状态 | 从不 | 已戒 | 偶尔 | 每天 |
|----------|------|------|------|------|
| COPD 率 | 3.58% | 12.78% | 15.86% | **20.59%** |

吸烟者患病率是从不吸烟者的 **5.8 倍**。

| 年龄段 | 18–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 |
|--------|-------|-------|-------|-------|-------|-------|
| COPD 率 | 2.04% | 3.07% | 5.11% | 9.24% | 11.14% | 13.20% |

## 特征重要性

`gen_health` 0.3961 > `smoker` 0.2488 > `depression` 0.0709 > `age` 0.0660 >
`income` 0.0410 > `hypertension` 0.0393 > …

> **解读警告**：
> 1. **吸烟第 2** —— 与病因学一致，是最主要的可干预危险因素。
> 2. **自评健康第 1** —— 主要来自**反向因果**（已患病者自评更差），不宜作因果解读。
> 3. **年龄仅第 4** —— 低于其真实流行病学地位，因其效应被吸烟等中介变量吸收。

## 泄漏防控

COPD 模块有一组**仅对 COPD 患者追问**的后续问题，全部排除以防标签泄漏：
`COPDCOGH`、`COPDFLEM`、`COPDBRTH`、`COPDBTST`、`COPDSMOK`。
另 `ASTHMA3`（哮喘）属同类呼吸系统疾病，亦排除。

## 重要提示

1. **横断面数据**：BRFSS 问的是"是否曾被医生告知患病"，本模型做的是
   **病例识别（已患病可能性）**，**不是**"未来 N 年发病风险"。
2. **未诊断人群**：COPD 诊断率偏低，"阴性"不等于没有 COPD。
3. **自报数据**：风险因素与结局均为受访者自报，存在回忆与报告偏倚。
4. **人群适用性**：基于美国成年居民（2017 年），迁移到其他国家和地区需重新验证。
5. **声明**：模型仅用于健康风险筛查辅助参考与教学演示，
   **不能替代专业医生的诊断与肺功能检查**。
