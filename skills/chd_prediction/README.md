# 冠心病（CHD）风险预测 Skill（chd_prediction）

基于**真实权威全人群数据集** BRFSS 2017（美国 CDC 全美电话调查）
训练 XGBoost 二分类模型的冠心病/心肌梗死风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 关键指标

| 项目 | 值 |
|------|-----|
| 数据集 | BRFSS 2017（美国 CDC，50 州及属地成年居民抽样） |
| **目标人群** | **全人群，不限性别、种族**（性别/种族在数据构建阶段即未提取） |
| 原始 / 建模记录 | 450,016 → **445,872** |
| 特征 | 15 项（年龄、体格、生活方式、疾病史） |
| 标签 | CDC 派生变量 `_MICHD`：曾被医生告知患有冠心病或心肌梗死 |
| 阳性事件 | **39,357**（患病率 8.83%） |
| 决策阈值 | **0.08**（训练集 CV 选定，非 0.5） |
| 留出测试集（89,175） | **AUC 0.8479**，**平衡准确率 0.7693**，敏感度 0.8389，特异度 0.6997 |
| 3×5 折交叉验证 | **AUC 0.8460 ± 0.0021**，**平衡准确率 0.7684 ± 0.0028** |
| 验收标准 | AUC ≥ 0.80、平衡准确率 ≥ 0.70 —— **全部达标** |

## ⚠️ 为什么不用"准确率"

本任务阳性率仅 8.83%，**多数类基线准确率就有 91.17%**。
本模型的普通准确率是 0.7120，**反而低于该基线** —— 这是不平衡分类中
准确率指标失效（accuracy paradox）的正常现象：把阈值调高、几乎不预测阳性，
准确率很容易"刷"到 91% 以上，但那样的模型毫无筛查价值。

因此验收采用 **AUC + 平衡准确率 + PR-AUC + 敏感度/特异度**。

## 目录结构

```
chd_prediction/
├── SKILL.md                              # skill 定义与完整使用说明
├── README.md                             # 本文件
├── requirements.txt
├── data/
│   ├── coronary_heart_disease_brfss.csv  # 445,872 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md                # 溯源、校验值、编码核对、构建规则、限制
├── scripts/
│   ├── fetch_data.py                     # 从 CDC 下载 + SHA256 校验 + 重建
│   ├── train_model.py                    # 训练 + 验证 + 保存全部参数
│   ├── predict.py                        # 单个体推理
│   ├── batch_predict.py                  # 批量推理
│   └── feature_importance.py             # 特征重要性
├── models/
│   ├── chd_xgb_model.json                # XGBoost 原生模型（推理首选）
│   ├── chd_xgb_model.ubj
│   ├── imputer.joblib
│   └── model_metadata.json               # 全部参数 / 指标 / 阈值 / 插补中位数
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
python scripts/predict.py --age 62 --bmi 29.5 --smoker 3 --diabetes 0 \
    --hypertension 1 --high-chol 1

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
# 划分：train_test_split(test_size=0.2, random_state=42, stratify=y) → 356,697/89,175
# 阈值：训练集 5 折 CV 最大化平衡准确率 → 0.08
```

完整记录见 `models/model_metadata.json`。

## 特征重要性

`hypertension` 0.3978 > `gen_health` 0.1740 > `high_chol` 0.1146 > `age` 0.1117 >
`copd` 0.0646 > `diabetes` 0.0438 > `smoker` 0.0296 > …

> **解读警告**：高血压、高胆固醇、自评健康居前，除真实病理关联外还含
> **检出偏倚/反向因果**（已确诊冠心病者被反复检查，更易查出高血压高胆固醇），
> 其重要性被高估，不可当作因果风险强度。

## 概率校准

留出测试集上预测概率分档与实测患病率高度吻合，因此输出概率可近似当**绝对风险**解读：

| 概率区间 | 实测冠心病率 |
|----------|--------------|
| 0.00–0.04 | 1.17% |
| 0.04–0.08 | 5.91% |
| 0.08–0.15 | 11.22% |
| 0.15–0.30 | 21.53% |
| ≥ 0.30 | 41.56% |

## 重要提示

1. **横断面数据**：BRFSS 问的是"是否曾被医生告知患病"，因此本模型做的是
   **病例识别（已患病可能性）**，**不是**"未来 N 年发病风险"，
   与 Framingham 类前瞻性风险模型含义不同，不可混用。
2. **主要为自报数据**：风险因素与结局均为受访者自报，存在回忆与报告偏倚。
3. **人群适用性**：基于美国成年居民（2017 年），迁移到其他国家和地区需重新验证。
4. **声明**：模型仅用于健康风险筛查辅助参考与教学演示，**不能替代专业医生的诊断**。
