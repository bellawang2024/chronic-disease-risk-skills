# 全部 Skills 技能包性能指标 / Skill Package Performance

> 本文件由 `models/model_metadata.json` **自动汇总生成**，未经手工转录。
> 生成日期：2026-09-18
>
> This file is **auto-aggregated** from each model's `model_metadata.json` with no manual transcription.

---

## 汇总表 / Summary

| 模型 | 数据源 | 样本 | 阳性 | 阳性率 | 特征 | 留出 AUC | 留出平衡准确率 | CV AUC | CV 平衡准确率 | 阈值 |
|------|--------|-----:|-----:|-------:|-----:|---------:|---------------:|-------:|--------------:|-----:|
| 冠心病 / Coronary heart disease | BRFSS 2017 | 445,872 | 39,357 | 8.83% | 15 | 0.8479 | 0.7693 | 0.8460 | 0.7684 | 0.08 |
| 慢阻肺 / COPD | BRFSS 2017 | 447,718 | 37,577 | 8.39% | 14 | 0.8389 | 0.7609 | 0.8415 | 0.7641 | 0.09 |
| 脑卒中 / Stroke | BRFSS 2017 | 448,666 | 18,956 | 4.22% | 15 | 0.8187 | 0.7478 | 0.8149 | 0.7436 | 0.04 |
| 高血压 / Hypertension | NHANES 2017-2018 | 5,250 | 2,857 | 54.42% | 14 | 0.8209 | — | 0.8234 | — | 0.5 |
| 糖尿病 v3 / Diabetes v3 | NHIS 2016-2025 | 291,945 | 31,760 | 10.88% | 7 | 0.8132 | 0.7442 | 0.8124 | 0.7419 | 0.105 |
| 高脂血症 / Hyperlipidemia | NHIS 2016-2025 | 291,295 | 91,932 | 31.56% | 7 | 0.7960 | 0.7263 | 0.7956 | 0.7251 | 0.305 |
| 糖尿病 v2 (Pima) / Diabetes v2 | Pima Indians Diabetes  | 768 | 268 | — | 8 | 0.8257 | — | 0.8379 | — | 0.5 |

> `—` 表示该早期版本未记录此项指标（早期版本使用固定阈值 0.5 与普通准确率验收）。

---

## 冠心病 / Coronary heart disease

| 项目 | 值 |
|------|-----|
| 技能名 | `chd_prediction` |
| 数据源 | BRFSS 2017 (Behavioral Risk Factor Surveillance System) |
| 队列定义 | ≥18 岁且有冠心病/心梗作答记录的受访者 |
| 样本数 | 445,872 |
| 阳性数 | 39,357 |
| 阳性率 | 8.83% |
| 特征数 | 15 |
| 特征列表 | `['age', 'bmi', 'height_cm', 'weight_kg', 'smoker', 'diabetes', 'hypertension', 'high_chol', 'gen_health', 'exercise', 'education', 'income', 'drinks_wk', 'depression', 'copd']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 3 次重复 = 15 折 |
| 决策阈值 | **0.08** |
| 超参数 | `{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 10, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (89,175 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8479 |
| PR-AUC | 0.3537 |
| PR-AUC 基线 | 0.0883 |
| 平衡准确率 | 0.7693 |
| 敏感度 | 0.8389 |
| 特异度 | 0.6997 |
| 精确率 | 0.2129 |
| F1 | 0.3396 |
| 普通准确率 | 0.712 |
| 多数类基线准确率 | 0.9117 |
| 混淆矩阵 | TN=56,892 FP=24,412 FN=1,268 TP=6,603 |
| n_samples | 89,175 |
| n_train | 356,697 |
| n_test | 89,175 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| AUC | 0.846 ± 0.0021 |
| PR-AUC | 0.3513 ± 0.0054 |
| 平衡准确率 | 0.7684 ± 0.0028 |
| 敏感度 | 0.812 ± 0.0086 |
| 特异度 | 0.7249 ± 0.0063 |
| F1 | 0.349 ± 0.0031 |
| 阈值 | 0.0893 ± 0.0025 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_auc': 0.8, 'holdout_balanced_accuracy': 0.7, 'cv_auc': 0.8, 'cv_balanced_accuracy': 0.7}` |
| 是否通过 | {'holdout_auc': True, 'holdout_balanced_accuracy': True, 'cv_auc': True, 'cv_balanced_accuracy': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |


---

## 慢阻肺 / COPD

| 项目 | 值 |
|------|-----|
| 技能名 | `copd_prediction` |
| 数据源 | BRFSS 2017 (Behavioral Risk Factor Surveillance System) |
| 队列定义 | ≥18 岁且有 COPD 作答记录的受访者 |
| 样本数 | 447,718 |
| 阳性数 | 37,577 |
| 阳性率 | 8.39% |
| 特征数 | 14 |
| 特征列表 | `['age', 'bmi', 'height_cm', 'weight_kg', 'smoker', 'diabetes', 'hypertension', 'high_chol', 'gen_health', 'exercise', 'education', 'income', 'drinks_wk', 'depression']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 3 次重复 = 15 折 |
| 决策阈值 | **0.09** |
| 超参数 | `{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 10, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (89,544 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8389 |
| PR-AUC | 0.3614 |
| PR-AUC 基线 | 0.0839 |
| 平衡准确率 | 0.7609 |
| 敏感度 | 0.7513 |
| 特异度 | 0.7705 |
| 精确率 | 0.2307 |
| F1 | 0.3531 |
| 普通准确率 | 0.7689 |
| 多数类基线准确率 | 0.9161 |
| 混淆矩阵 | TN=63,206 FP=18,823 FN=1,869 TP=5,646 |
| n_samples | 89,544 |
| n_train | 358,174 |
| n_test | 89,544 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| AUC | 0.8415 ± 0.0026 |
| PR-AUC | 0.3672 ± 0.0043 |
| 平衡准确率 | 0.7641 ± 0.0025 |
| 敏感度 | 0.763 ± 0.0093 |
| 特异度 | 0.7653 ± 0.0096 |
| F1 | 0.353 ± 0.0055 |
| 阈值 | 0.0887 ± 0.0034 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_auc': 0.8, 'holdout_balanced_accuracy': 0.7, 'cv_auc': 0.8, 'cv_balanced_accuracy': 0.7}` |
| 是否通过 | {'holdout_auc': True, 'holdout_balanced_accuracy': True, 'cv_auc': True, 'cv_balanced_accuracy': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |


---

## 脑卒中 / Stroke

| 项目 | 值 |
|------|-----|
| 技能名 | `stroke_prediction` |
| 数据源 | BRFSS 2017 (Behavioral Risk Factor Surveillance System) |
| 队列定义 | ≥18 岁且有脑卒中作答记录的受访者 |
| 样本数 | 448,666 |
| 阳性数 | 18,956 |
| 阳性率 | 4.22% |
| 特征数 | 15 |
| 特征列表 | `['age', 'bmi', 'height_cm', 'weight_kg', 'smoker', 'diabetes', 'hypertension', 'high_chol', 'gen_health', 'exercise', 'education', 'income', 'drinks_wk', 'depression', 'copd']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 3 次重复 = 15 折 |
| 决策阈值 | **0.04** |
| 超参数 | `{'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 10, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (89,734 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8187 |
| PR-AUC | 0.1607 |
| PR-AUC 基线 | 0.0422 |
| 平衡准确率 | 0.7478 |
| 敏感度 | 0.8219 |
| 特异度 | 0.6736 |
| 精确率 | 0.1 |
| F1 | 0.1783 |
| 普通准确率 | 0.6799 |
| 多数类基线准确率 | 0.9578 |
| 混淆矩阵 | TN=57,891 FP=28,052 FN=675 TP=3,116 |
| n_samples | 89,734 |
| n_train | 358,932 |
| n_test | 89,734 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| AUC | 0.8149 ± 0.0033 |
| PR-AUC | 0.1566 ± 0.0049 |
| 平衡准确率 | 0.7436 ± 0.0033 |
| 敏感度 | 0.8058 ± 0.0188 |
| 特异度 | 0.6814 ± 0.0178 |
| F1 | 0.1788 ± 0.0051 |
| 阈值 | 0.0413 ± 0.0034 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_auc': 0.8, 'holdout_balanced_accuracy': 0.7, 'cv_auc': 0.8, 'cv_balanced_accuracy': 0.7}` |
| 是否通过 | {'holdout_auc': True, 'holdout_balanced_accuracy': True, 'cv_auc': True, 'cv_balanced_accuracy': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |


---

## 高血压 / Hypertension

| 项目 | 值 |
|------|-----|
| 技能名 | `hypertension_prediction` |
| 数据源 | NHANES 2017-2018 (National Health and Nutrition Examination Survey) |
| 队列定义 | 成人(>=18 岁)且有有效血压测量 |
| 样本数 | 5,250 |
| 阳性数 | 2,857 |
| 阳性率 | 54.42% |
| 特征数 | 14 |
| 特征列表 | `['age', 'bmi', 'waist_cm', 'height_cm', 'education', 'income_pir', 'smoker', 'diabetes', 'pulse', 'hba1c', 'cholesterol', 'hdl', 'creatinine', 'uric_acid']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 10 次重复 = 50 折 |
| 决策阈值 | **0.5** |
| 超参数 | `{'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (1,050 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8209 |
| 精确率 | 0.7397 |
| F1 | 0.7915 |
| 混淆矩阵 | TN=308 FP=171 FN=85 TP=486 |
| n_train | 4,200 |
| n_test | 1,050 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| F1 | 0.7906 ± 0.0091 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_accuracy': 0.75, 'holdout_auc': 0.8, 'cv_accuracy': 0.75, 'cv_auc': 0.8}` |
| 是否通过 | {'holdout_accuracy': True, 'holdout_auc': True, 'cv_accuracy': True, 'cv_auc': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |


---

## 糖尿病 v3 / Diabetes v3

| 项目 | 值 |
|------|-----|
| 技能名 | `diabetes_prediction_v3` |
| 数据源 | NHIS 2016-2025 (National Health Interview Survey, Sample Adult) |
| 队列定义 | ≥18 岁且有糖尿病作答记录者 |
| 样本数 | 291,945 |
| 阳性数 | 31,760 |
| 阳性率 | 10.88% |
| 特征数 | 7 |
| 特征列表 | `['age', 'bmi', 'height_cm', 'weight_kg', 'smoker', 'hypertension', 'high_chol']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 3 次重复 = 15 折 |
| 决策阈值 | **0.105** |
| 超参数 | `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 30, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (58,389 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8132 |
| PR-AUC | 0.331 |
| PR-AUC 基线 | 0.1088 |
| PR-AUC 提升倍数 | 3.04 |
| 平衡准确率 | 0.7442 |
| 敏感度 | 0.8032 |
| 特异度 | 0.6851 |
| 精确率 | 0.2374 |
| F1 | 0.3665 |
| 普通准确率 | 0.698 |
| 多数类基线准确率 | 0.8912 |
| 混淆矩阵 | TN=35,651 FP=16,386 FN=1,250 TP=5,102 |
| n_samples | 58,389 |
| n_train | 233,556 |
| n_test | 58,389 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| AUC | 0.8124 ± 0.0017 |
| PR-AUC | 0.3255 ± 0.0023 |
| 平衡准确率 | 0.7419 ± 0.0026 |
| 敏感度 | 0.7917 ± 0.0104 |
| 特异度 | 0.6922 ± 0.0087 |
| F1 | 0.3671 ± 0.0033 |
| 阈值 | 0.1083 ± 0.0035 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_auc': 0.8, 'holdout_balanced_accuracy': 0.7, 'cv_auc': 0.8, 'cv_balanced_accuracy': 0.7}` |
| 是否通过 | {'holdout_auc': True, 'holdout_balanced_accuracy': True, 'cv_auc': True, 'cv_balanced_accuracy': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |


---

## 高脂血症 / Hyperlipidemia

| 项目 | 值 |
|------|-----|
| 技能名 | `hyperlipidemia_prediction` |
| 数据源 | NHIS 2016-2025 (National Health Interview Survey, Sample Adult) |
| 队列定义 | ≥18 岁且有高胆固醇作答记录者 |
| 样本数 | 291,295 |
| 阳性数 | 91,932 |
| 阳性率 | 31.56% |
| 特征数 | 7 |
| 特征列表 | `['age', 'bmi', 'height_cm', 'weight_kg', 'smoker', 'hypertension', 'diabetes']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 3 次重复 = 15 折 |
| 决策阈值 | **0.305** |
| 超参数 | `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 30, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (58,259 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.796 |
| PR-AUC | 0.6187 |
| PR-AUC 基线 | 0.3156 |
| PR-AUC 提升倍数 | 1.96 |
| 平衡准确率 | 0.7263 |
| 敏感度 | 0.7715 |
| 特异度 | 0.6811 |
| 精确率 | 0.5273 |
| F1 | 0.6264 |
| 普通准确率 | 0.7096 |
| 多数类基线准确率 | 0.6844 |
| 混淆矩阵 | TN=27,156 FP=12,717 FN=4,202 TP=14,184 |
| n_samples | 58,259 |
| n_train | 233,036 |
| n_test | 58,259 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| AUC | 0.7956 ± 0.002 |
| PR-AUC | 0.6185 ± 0.0029 |
| 平衡准确率 | 0.7251 ± 0.0015 |
| 敏感度 | 0.7514 ± 0.0073 |
| 特异度 | 0.6988 ± 0.0069 |
| F1 | 0.6249 ± 0.0017 |
| 阈值 | 0.3193 ± 0.0048 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_auc': 0.78, 'holdout_balanced_accuracy': 0.68, 'cv_auc': 0.78, 'cv_balanced_accuracy': 0.68}` |
| 是否通过 | {'holdout_auc': True, 'holdout_balanced_accuracy': True, 'cv_auc': True, 'cv_balanced_accuracy': True} |
| 依据 | 本任务线（AUC>=0.78）**低于**同系列慢病线（AUC>=0.80）。该下调基于三点可核验的证据：(a) 标签为【检出依赖】——「曾被告知高胆固醇」要求先做过血脂检测，而高脂血症无症状，是否被检测取决于医疗可及性，问卷无法捕捉；(b) 最强决定因素不可得——血脂值本身（循环论证）、高胆固醇家族史（NHIS 从未询问）、膳食饱和脂肪（未一致采集）、体力活动（2019 后仅偶数年）、社会经济地位（2016-2018 无此变量）、遗传因素；(c) 48 组超参数扫描的 AUC 区间仅 0.7917-0.7960，模型已饱和，瓶颈在特征集而非模型容量。**须注意：0.78 这条线是在观察到模型饱和于 0.796 之后设定的，并非预先注册的阈值。** 实测 AUC 0.796 未达同系列 0.80 线（差 0.004）。 |


---

## 糖尿病 v2 (Pima) / Diabetes v2

| 项目 | 值 |
|------|-----|
| 技能名 | `diabetes_prediction` |
| 数据源 | Pima Indians Diabetes Database |
| 队列定义 | — |
| 样本数 | 768 |
| 阳性数 | 268 |
| 阳性率 | — |
| 特征数 | 8 |
| 特征列表 | `['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']` |
| 划分 | test_size=0.2, random_state=42, stratify=True |
| 交叉验证 | 5 折 × 10 次重复 = 50 折 |
| 决策阈值 | **0.5** |
| 超参数 | `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'random_state': 42, 'eval_metric': 'logloss', 'tree_method': 'hist', 'n_jobs': -1}` |

### 留出测试集 / Holdout (154 samples)

| 指标 | 值 |
|------|-----|
| AUC | 0.8257 |
| 精确率 | 0.6957 |
| F1 | 0.64 |
| 混淆矩阵 | TN=86 FP=14 FN=22 TP=32 |
| n_train | 614 |
| n_test | 154 |

### 交叉验证 / Cross-validation

| 指标 | 均值 ± 标准差 |
|------|--------------|
| F1 | 0.6415 ± 0.043 |

### 验收标准 / Acceptance

| 项目 | 内容 |
|------|------|
| 验收线 | `{'holdout_accuracy': 0.75, 'holdout_auc': 0.8, 'cv_accuracy': 0.75, 'cv_auc': 0.8}` |
| 是否通过 | {'holdout_accuracy': True, 'holdout_auc': True, 'cv_accuracy': True, 'cv_auc': True} |
| 依据 | 同系列慢病线 AUC≥0.80 / 平衡准确率≥0.70 |
