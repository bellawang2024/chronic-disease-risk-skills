# 脑卒中（Stroke）风险预测 Skill（stroke_prediction）

基于**真实权威全人群数据集** BRFSS 2017（美国 CDC 全美电话调查）
训练 XGBoost 二分类模型的脑卒中风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 关键指标

| 项目 | 值 |
|------|-----|
| 数据集 | BRFSS 2017（美国 CDC，50 州及属地成年居民抽样） |
| **目标人群** | **全人群，不限性别、种族**（性别/种族在数据构建阶段即未提取） |
| 原始 / 建模记录 | 450,016 → **448,666** |
| 特征 | 15 项（年龄、体格、生活方式、疾病史） |
| 标签 | BRFSS 核心变量 `CVDSTRK3`：曾被医生告知患有脑卒中 |
| 阳性事件 | **18,956**（患病率 4.22%） |
| 决策阈值 | **0.04**（训练集 CV 选定，非 0.5） |
| 留出测试集（89,734） | **AUC 0.8187**，**平衡准确率 0.7478**，敏感度 0.8219，特异度 0.6736 |
| 3×5 折交叉验证 | **AUC 0.8149 ± 0.0033**，**平衡准确率 0.7436 ± 0.0033** |
| 验收标准 | AUC ≥ 0.80、平衡准确率 ≥ 0.70 —— **全部达标** |

## ⚠️ 为什么不用"准确率"

本任务阳性率仅 4.22%，**多数类基线准确率就有 95.78%**。
本模型的普通准确率是 0.6799，**反而低于该基线** —— 这是不平衡分类中
准确率指标失效（accuracy paradox）的正常现象：把阈值调高、几乎不预测阳性，
准确率很容易"刷"到 95% 以上，但那样的模型毫无筛查价值。

因此验收采用 **AUC + 平衡准确率 + PR-AUC + 敏感度/特异度**。

## ⚠️ 关键危险因素缺失

心房颤动、颈动脉狭窄、抗凝治疗、既往 TIA 等**卒中核心危险因素不在数据中**
（BRFSS 核心问卷未覆盖）。因此本模型**不应被当作完整的卒中风险评分**，
只能作为基于常规可获取指标的粗略风险分层工具。这也是它 AUC（0.82）
低于同系列冠心病模型（0.85）的主要原因。

## 目录结构

```
stroke_prediction/
├── SKILL.md                        # skill 定义与完整使用说明
├── README.md                       # 本文件
├── requirements.txt
├── data/
│   ├── stroke_brfss.csv            # 448,666 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md          # 溯源、校验值、编码核对、构建规则、限制
├── scripts/
│   ├── fetch_data.py               # 从 CDC 下载 + SHA256 校验 + 重建
│   ├── train_model.py              # 训练 + 验证 + 保存全部参数
│   ├── predict.py                  # 单个体推理
│   ├── batch_predict.py            # 批量推理
│   └── feature_importance.py       # 特征重要性
├── models/
│   ├── stroke_xgb_model.json       # XGBoost 原生模型（推理首选）
│   ├── stroke_xgb_model.ubj
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
python scripts/predict.py --age 68 --bmi 28.0 --smoker 3 --diabetes 1 \
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
XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
              subsample=0.9, colsample_bytree=0.9, min_child_weight=10,
              random_state=42, eval_metric="logloss", tree_method="hist")
# 预处理：SimpleImputer(strategy="median")，无需标准化
# 划分：train_test_split(test_size=0.2, random_state=42, stratify=y) → 358,932/89,734
# 阈值：训练集 5 折 CV 最大化平衡准确率 → 0.04
```

完整记录见 `models/model_metadata.json`。

## 数据质量旁证

卒中率随年龄**严格单调上升**，与流行病学完全一致：

| 年龄段 | 18–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 |
|--------|-------|-------|-------|-------|-------|-------|
| 卒中率 | 0.49% | 0.98% | 2.24% | 3.86% | 5.36% | 8.45% |

## 特征重要性

`hypertension` 0.3734 > `gen_health` 0.2335 > `age` 0.1178 > `high_chol` 0.0536 >
`income` 0.0467 > `copd` 0.0373 > …

> **解读警告**：高血压与自评健康居前两位，除真实病理关联外还含
> **检出偏倚/反向因果**（已确诊卒中者被反复检查，更易查出高血压），
> 其重要性被高估。另外**年龄仅排第 3**，低于其真实流行病学地位 ——
> 反映的是"高血压"这一中介变量吸收了部分年龄效应。

## 概率校准

留出测试集上预测概率分档与实测患病率高度吻合，输出概率可近似当**绝对风险**解读：

| 概率区间 | 实测卒中率 |
|----------|------------|
| 0.00–0.02 | 0.64% |
| 0.02–0.04 | 2.84% |
| 0.04–0.08 | 6.06% |
| 0.08–0.15 | 11.04% |
| 0.15–0.30 | 19.13% |
| ≥ 0.30 | 35.38% |

## 重要提示

1. **横断面数据**：BRFSS 问的是"是否曾被医生告知患病"，因此本模型做的是
   **病例识别（已患病可能性）**，**不是**"未来 N 年发病风险"。
2. **关键因素缺失**：心房颤动、颈动脉狭窄等未覆盖，不可当作完整卒中风险评分。
3. **自报数据**：卒中自报准确性低于其他诊断，误分类削弱性能上限。
4. **人群适用性**：基于美国成年居民（2017 年），迁移到其他国家和地区需重新验证。
5. **声明**：模型仅用于健康风险筛查辅助参考与教学演示，**不能替代专业医生的诊断**。
