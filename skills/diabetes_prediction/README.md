# 糖尿病风险预测 Skill（diabetes_prediction）

基于**真实权威公开数据集** Pima Indians Diabetes Database（NIDDK，OpenML id 37）
训练 XGBoost 二分类模型的糖尿病风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 关键指标

| 项目 | 值 |
|------|-----|
| 数据集 | Pima Indians Diabetes Database（768 真实临床样本，8 特征） |
| 数据校验 | MD5 `3cbaa3e54586aa88cf6aacb4033e4470`（与 OpenML 官方公布值一致），并已双源交叉验证 |
| 留出测试集（154 样本） | 准确率 **0.7662**，AUC **0.8257** |
| 10×5 折交叉验证 | 准确率 **0.7629 ± 0.0253**，AUC **0.8379 ± 0.0264** |
| 验收标准 | 准确率 ≥ 0.75、AUC ≥ 0.80 —— **全部达标** |

## 目录结构

```
diabetes_prediction/
├── SKILL.md                       # skill 定义与完整使用说明
├── README.md                      # 本文件
├── requirements.txt               # 依赖
├── data/
│   ├── pima_indians_diabetes.csv  # 768 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md         # 数据溯源、校验值、引用、使用限制
├── scripts/
│   ├── fetch_data.py              # 重新下载数据（含 MD5 校验）
│   ├── train_model.py             # 训练 + 验证 + 保存全部参数
│   ├── predict.py                 # 单患者推理（CLI / 可导入函数）
│   ├── batch_predict.py           # 批量推理（CSV/JSON）
│   └── feature_importance.py      # 特征重要性查看与绘图
├── models/                        # 推理模型工件（已训练好）
│   ├── diabetes_xgb_model.json    # XGBoost 原生模型（推理首选）
│   ├── diabetes_xgb_model.ubj     # 二进制备用格式
│   ├── imputer.joblib             # 中位数插补器
│   └── model_metadata.json        # 全部参数 / 指标 / 插补中位数 / 环境版本
├── results/
│   ├── holdout_metrics.json       # 留出测试集指标
│   ├── cv_metrics.json            # 交叉验证指标
│   ├── feature_importance.json    # 特征重要性
│   └── batch_results.csv          # 批量预测示例输出
└── examples/
    └── patients.csv               # 示例输入
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 单患者预测（模型已内置，无需先训练）
python scripts/predict.py --preg 2 --plas 120 --pres 70 --skin 25 \
    --insu 150 --mass 30 --pedi 0.5 --age 35

# 3. 批量预测
python scripts/batch_predict.py --input examples/patients.csv --output results/batch_results.csv

# 4.（可选）重新训练并验证
python scripts/train_model.py

# 5.（可选）校验内置数据集
python scripts/fetch_data.py --check
```

## 训练参数（已完整记录）

```python
XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
              random_state=42, eval_metric="logloss", tree_method="hist")
# 预处理：plas/pres/skin/insu/mass 中的 0 视为缺失，中位数插补
# 划分：train_test_split(test_size=0.2, random_state=42, stratify=y) → 614/154
# 无需标准化（XGBoost 对尺度不敏感）
```

插补中位数：plas=117.0, pres=72.0, skin=29.0, insu=125.0, mass=32.3

完整记录见 `models/model_metadata.json`。

## 特征重要性

`plas`（血糖）0.2800 > `mass`（BMI）0.1610 > `age`（年龄）0.1447 > `insu` 0.0993 >
`pedi` 0.0909 > `preg` 0.0878 > `skin` 0.0786 > `pres` 0.0577

血糖排在首位，与糖尿病病理机制一致。

## 重要提示

1. **人群限制**：数据集全部为 ≥21 岁的 Pima 印第安女性，模型**不可直接外推**到
   男性、儿童、其他族裔或地区人群。
2. **召回率**：阈值 0.5 下召回率约 0.59（存在漏筛）。筛查场景建议下调阈值，
   详见 `SKILL.md` 注意事项。
3. **声明**：本模型仅用于健康风险筛查辅助参考与教学演示，**不能替代专业医生的诊断**。
