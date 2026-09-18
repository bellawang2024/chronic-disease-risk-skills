# 高血压风险预测 Skill（hypertension_prediction）

基于**真实权威全人群数据集** NHANES 2017-2018（美国 CDC / NCHS 全国代表性抽样）
训练 XGBoost 二分类模型的高血压风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 关键指标

| 项目 | 值 |
|------|-----|
| 数据集 | NHANES 2017-2018（美国 CDC/NCHS，全国代表性抽样） |
| **目标人群** | **美国非机构化平民人口，不限性别、种族** （男 2,549 / 女 2,701） |
| 样本 | 5,250 名成人（≥18 岁，有有效血压测量） |
| 特征 | 14 项常规体检/问卷/血检指标，**不含性别、种族** |
| 标签 | 高血压（ACC/AHA 2017：SBP≥130 或 DBP≥80 或服降压药），患病率 54.4% |
| 留出测试集（1,050） | 准确率 **0.7562**，AUC **0.8209** |
| 10×5 折交叉验证 | 准确率 **0.7597 ± 0.0106**，AUC **0.8234 ± 0.0113** |
| 验收标准 | 准确率 ≥ 0.75、AUC ≥ 0.80 —— **全部达标** |

## 目录结构

```
hypertension_prediction/
├── SKILL.md                              # skill 定义与完整使用说明
├── README.md                             # 本文件
├── requirements.txt                      # 依赖
├── data/
│   ├── hypertension_nhanes.csv           # 5,250 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md                # 溯源、校验值、构建规则、使用限制
├── scripts/
│   ├── fetch_data.py                     # 从 CDC 下载 + SHA256 校验 + 重建数据集
│   ├── train_model.py                    # 训练 + 验证 + 保存全部参数
│   ├── predict.py                        # 单个体推理（CLI / 可导入函数）
│   ├── batch_predict.py                  # 批量推理（CSV/JSON）
│   └── feature_importance.py             # 特征重要性查看与绘图
├── models/                               # 推理模型工件（已训练好）
│   ├── hypertension_xgb_model.json       # XGBoost 原生模型（推理首选）
│   ├── hypertension_xgb_model.ubj        # 二进制备用格式
│   ├── imputer.joblib                    # 中位数插补器
│   └── model_metadata.json               # 全部参数 / 指标 / 插补中位数 / 环境版本
├── results/
│   ├── holdout_metrics.json
│   ├── cv_metrics.json
│   ├── feature_importance.json
│   └── batch_results.csv
└── examples/
    └── people.csv                        # 示例输入
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 个体预测（只需 3 个核心指标，模型已内置）
python scripts/predict.py --age 55 --bmi 27.5 --waist 95

# 3. 批量预测
python scripts/batch_predict.py --input examples/people.csv --output results/batch_results.csv

# 4.（可选）重新训练并验证
python scripts/train_model.py

# 5.（可选）从 CDC 重新下载并校验数据
python scripts/fetch_data.py
```

## 训练参数（已完整记录）

```python
XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
              random_state=42, eval_metric="logloss", tree_method="hist")
# 预处理：SimpleImputer(strategy="median")，无需标准化
# 划分：train_test_split(test_size=0.2, random_state=42, stratify=y) → 4200/1050
```

完整记录见 `models/model_metadata.json`。

## 特征重要性

`age`（年龄）0.3129 > `diabetes`（糖尿病）0.1010 > `hba1c`（糖化血红蛋白）0.1001 >
`waist_cm`（腰围）0.0942 > `uric_acid`（尿酸）0.0591 > `bmi` 0.0476 > …

年龄居首，其次为糖代谢异常与中心性肥胖，与高血压流行病学认识一致。

## 为什么默认不用性别/种族

| 特征方案 | 特征数 | CV AUC | CV 准确率 |
|----------|--------|--------|-----------|
| **默认（不含性别/种族）** | **14** | **0.8234** | **0.7597** |
| 含性别/种族 | 16 | 0.8301 | 0.7594 |

纳入性别与种族仅提升 0.007 AUC，准确率基本持平，因此默认模型不使用任何人群
划分属性 —— 适用人群更广，也避免对特定群体产生差异化输出。
如需纳入：`python scripts/train_model.py --include-sensitive`

## 重要提示

1. **人群适用性**：数据来自美国非机构化平民人口（2017–2018），
   迁移到其他国家和地区人群时需重新验证。
2. **仅成人**：队列只含 ≥18 岁成人。
3. **精度上限**：标签源自单次访视实测血压，存在测量波动，
   文献中同类模型 AUC 多在 0.80–0.85，本模型处于该区间。
4. **声明**：模型仅用于健康风险筛查辅助参考与教学演示，**不能替代专业医生的诊断**。
