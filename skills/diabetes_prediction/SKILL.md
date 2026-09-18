---
name: diabetes_prediction
description: "糖尿病风险预测模型 - 基于真实权威数据集 Pima Indians Diabetes Database（NIDDK/OpenML，768 样本 8 特征）训练 XGBoost 二分类模型，留出测试集准确率 0.7662、AUC 0.8257。触发条件：用户提到糖尿病预测、糖尿病风险评估、血糖预测、健康风险分析等。"
version: 1.1.0
license: 仅供学习研究使用（数据为 OpenML 公开数据）
metadata:
  display_name: 糖尿病（Diabetes）风险预测
  category: 技能开发
---

> **Important:** All `scripts/` paths are relative to this skill directory.
> Run with: `cd {this_skill_dir} && python scripts/...`
> Or use the `cwd` parameter of `execute_shell_command`.

# 糖尿病风险预测模型

## 概述

基于**真实、可追溯的公开权威数据集** Pima Indians Diabetes Database，
使用 XGBoost 训练糖尿病风险二分类模型。本 skill 内置**已训练好的推理模型**，
解压即可预测，无需重新训练。

| 项目 | 值 |
|------|-----|
| 任务 | 二分类（是否患糖尿病，WHO 标准判定） |
| 算法 | XGBoost (`xgboost.XGBClassifier`) |
| 数据集 | Pima Indians Diabetes Database（NIDDK 原始所有，OpenML id 37 分发） |
| 样本数 | **768**（全部真实临床记录，非模拟数据） |
| 特征数 | **8**（简洁的常规临床指标） |
| 类别分布 | 阳性 268 / 阴性 500（阳性率 34.9%） |
| 划分 | test_size=0.2, random_state=42, stratify=True → 614 训练 / 154 测试 |
| 决策阈值 | 0.5（经训练集交叉验证确认最优） |

## 数据权威性与可追溯性

数据来源链条完整，且经过**双源交叉验证**：

| 环节 | 内容 |
|------|------|
| 原始所有者 | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA |
| 数据捐赠者 | Vincent Sigillito, Johns Hopkins University, Applied Physics Laboratory |
| 数据接收日期 | 1990-05-09 |
| 分发渠道 | OpenML dataset id 37，licence: **Public** |
| 下载地址 | https://openml.org/data/v1/download/37/diabetes.arff |
| 引用文献 | Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proc. Symposium on Computer Applications in Medical Care, 261–265. |

**校验值**（可用 `python scripts/fetch_data.py --check` 复验）：

| 文件 | 校验值 |
|------|--------|
| OpenML 原始 ARFF（MD5，与官方 API 公布值一致） | `3cbaa3e54586aa88cf6aacb4033e4470` |
| OpenML 原始 ARFF（SHA256） | `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` |
| `data/pima_indians_diabetes.csv`（内置） | `fa9f4490a5e0d9eb7a0a65e758388737ac74f3828d4c877acc6145abcbbb2997` |

构建时还从**独立镜像**下载了同一数据集逐值比对：768 条记录全部数值完全一致。
详见 `data/DATA_PROVENANCE.md`。

## 依赖

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

可选（仅绘图时需要）：`pip install matplotlib`

## 功能

### 1. 预测单个患者（模型已内置，无需先训练）

```bash
python scripts/predict.py --preg 2 --plas 120 --pres 70 --skin 25 --insu 150 --mass 30 --pedi 0.5 --age 35
```

加 `--json` 可获得结构化输出。

参数说明：

| 参数 | 描述 | 单位 | 合理范围 |
|------|------|------|----------|
| preg | 怀孕次数 | 次 | 0–20 |
| plas | 口服糖耐量试验 2 小时血糖浓度 | mg/dL | 0–300 |
| pres | 舒张压 | mm Hg | 0–200 |
| skin | 三头肌皮褶厚度 | mm | 0–120 |
| insu | 2 小时血清胰岛素 | mu U/ml | 0–1000 |
| mass | BMI 体重指数 | kg/m² | 0–80 |
| pedi | 糖尿病家族遗传函数 | — | 0–3 |
| age | 年龄 | 岁 | 1–120 |

> `plas` / `pres` / `skin` / `insu` / `mass` 中传入 **0 表示缺失**，
> 推理时会自动用训练中位数插补，并在输出中标注 `imputed_features`。

### 2. 批量预测

```bash
python scripts/batch_predict.py --input examples/patients.csv --output results.csv
```

输入 CSV（列顺序无关，兼容 Pima 原始列名 `Glucose` / `BMI` / `Outcome` 等）：

```csv
preg,plas,pres,skin,insu,mass,pedi,age
2,120,70,25,150,30,0.5,35
1,100,60,20,80,25,0.3,30
```

### 3. 训练 / 重新验证模型

```bash
python scripts/train_model.py
```

脚本会依次执行：留出测试集评估 → 10×5 折重复交叉验证 → 验收标准检查 →
用全量 768 条样本重训发布模型，并把全部参数与指标写入 `models/model_metadata.json`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/pima_indians_diabetes.csv` | 外部训练 CSV |
| `--quick` | 关闭 | 减少交叉验证重复次数，加快运行 |
| `--quiet` | 关闭 | 静默模式 |

未达验收标准时脚本以退出码 2 结束。

### 4. 特征重要性

```bash
python scripts/feature_importance.py
python scripts/feature_importance.py --plot results/feature_importance.png
```

### 5. 数据下载与校验

```bash
python scripts/fetch_data.py           # 重新下载并校验 MD5
python scripts/fetch_data.py --check   # 仅校验内置数据
```

### 6. Python 代码调用

```python
import sys
sys.path.insert(0, "scripts")
from predict import predict_patient

result = predict_patient(preg=2, plas=120, pres=70, skin=25,
                         insu=150, mass=30, pedi=0.5, age=35)
print(result["probability_pct"], result["label"], result["risk_level"])
# 49.18% 非糖尿病 中等风险
```

## 训练参数（已记录，可复现）

```python
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
)
```

超参数由**训练集内层 5 折 GridSearchCV 按 AUC 选出**（未使用测试集调参）。

预处理：`plas/pres/skin/insu/mass` 中的 0 视为缺失 → `SimpleImputer(strategy="median")`。
XGBoost 对特征尺度不敏感，**不需要标准化**。插补中位数：

| 列 | plas | pres | skin | insu | mass |
|----|------|------|------|------|------|
| 中位数 | 117.0 | 72.0 | 29.0 | 125.0 | 32.3 |

完整参数、指标、特征顺序、环境版本均记录在 `models/model_metadata.json`。

## 模型评估结果（实测）

### 留出测试集（154 样本，未参与训练与调参）

| 指标 | 数值 | 验收标准 | 结论 |
|------|------|----------|------|
| 准确率 | **0.7662** | ≥ 0.75 | 通过 |
| AUC | **0.8257** | ≥ 0.80 | 通过 |
| 精确率 | 0.6957 | — | — |
| 召回率 | 0.5926 | — | — |
| F1 分数 | 0.6400 | — | — |

混淆矩阵：TN=86, FP=14, FN=22, TP=32

### 10×5 折重复交叉验证（全量 768 样本，50 折）

| 指标 | 均值 ± 标准差 | 验收标准 | 结论 |
|------|---------------|----------|------|
| AUC | **0.8379 ± 0.0264** | ≥ 0.80 | 通过 |
| 准确率 | **0.7629 ± 0.0253** | ≥ 0.75 | 通过 |
| 精确率 | 0.6798 ± 0.0474 | — | — |
| 召回率 | 0.6099 ± 0.0565 | — | — |
| F1 | 0.6415 ± 0.0430 | — | — |

上述水平与 Pima 数据集公开文献基准（准确率约 76%–78%，AUC 约 0.82–0.83）一致。

> **关于发布模型**：上表指标来自 614/154 划分与交叉验证；
> 随 skill 发布的模型用**全量 768 条**样本重训（生产惯例，数据更多效果更好）。
> 因此发布模型在训练集上的自检值为 AUC 0.9291 / 准确率 0.8503，
> 该数值**不是**泛化性能，请以留出测试集与交叉验证指标为准。

## 特征重要性（实测，临床合理）

| 排序 | 特征 | 中文名 | 重要性 |
|------|------|--------|--------|
| 1 | plas | 口服糖耐量试验2小时血糖浓度 | 0.2800 |
| 2 | mass | BMI体重指数 | 0.1610 |
| 3 | age | 年龄 | 0.1447 |
| 4 | insu | 2小时血清胰岛素 | 0.0993 |
| 5 | pedi | 糖尿病家族遗传函数 | 0.0909 |
| 6 | preg | 怀孕次数 | 0.0878 |
| 7 | skin | 三头肌皮褶厚度 | 0.0786 |
| 8 | pres | 舒张压 | 0.0577 |

血糖浓度排在首位，与糖尿病病理机制一致，可作为模型合理性的旁证。

## 推理模型工件

| 文件 | 说明 |
|------|------|
| `models/diabetes_xgb_model.json` | XGBoost 原生模型（**推理首选**，跨版本可移植） |
| `models/diabetes_xgb_model.ubj` | XGBoost 二进制模型（等价备用） |
| `models/imputer.joblib` | 中位数插补器（备用） |
| `models/model_metadata.json` | 全部训练参数、指标、插补中位数、特征顺序、环境版本 |
| `results/holdout_metrics.json` | 留出测试集指标 |
| `results/cv_metrics.json` | 交叉验证指标 |
| `results/feature_importance.json` | 特征重要性 |
| `data/pima_indians_diabetes.csv` | 内置训练数据（768 条） |
| `data/DATA_PROVENANCE.md` | 数据溯源与校验说明 |

推理时**特征顺序固定为** `preg, plas, pres, skin, insu, mass, pedi, age`。

推理只需 XGBoost + numpy：即使 `joblib` / `imputer.joblib` 缺失，
也能仅凭 `model_metadata.json` 中记录的插补中位数完成预测，结果完全一致
（已实测：手工插补与 `SimpleImputer` 在 768×8 = 6144 个值上**逐值完全一致**）。

## 输出结果示例

```json
{
  "probability": 0.4918,
  "probability_pct": "49.18%",
  "prediction": 0,
  "label": "非糖尿病",
  "risk_level": "中等风险",
  "advice": "建议控制饮食与体重、增加运动，并定期复查血糖",
  "decision_threshold": 0.5,
  "imputed_features": []
}
```

风险等级划分：

- 风险概率 < 30%：低风险
- 风险概率 30%–70%：中等风险
- 风险概率 > 70%：高风险

## 注意事项

1. **人群限制（重要）**：数据集全部为 **≥21 岁的 Pima 印第安女性**，
   模型结论**不可直接外推**到男性、儿童、其他族裔或地区人群。
2. **数据年代**：数据采集于 1990 年前后，反映当时的临床测量与诊断标准。
3. **召回率偏低**：阈值 0.5 下召回率约 0.59，即存在漏筛。
   若用于筛查场景需提高灵敏度，可下调阈值（如 0.35，召回率可提升至约 0.75），
   代价是假阳性增多 — 阈值应结合具体应用场景选择。
4. **单调性细节**：血糖在 180→200 mg/dL 区间，固定其他特征时风险预测值
   有 0.0038 的微降（树模型的正常现象，该区间两者均已是高风险），不影响使用。
5. **临床决策**：本模型仅作为健康风险筛查的辅助参考与教学演示，
   **不能替代专业医生的诊断**。
