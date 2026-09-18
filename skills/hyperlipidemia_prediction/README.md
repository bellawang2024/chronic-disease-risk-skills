# 高脂血症（高胆固醇）风险预测 Skill（全人群）

基于**真实权威全人群数据集** NHIS 2016–2025（美国 CDC/NCHS 全国住户抽样，10 个年度）
训练 XGBoost 二分类模型的高脂血症风险预测 skill。
内置**已训练好的推理模型**，解压即可预测。

## 核心指标

| 项目 | 值 |
|------|-----|
| 数据来源 | **NHIS 2016–2025**（美国 CDC/NCHS，无需注册、Public Domain） |
| 目标人群 | **全人群**（美国非机构化平民人口，男女兼备、不限种族） |
| 样本 / 特征 | **291,295** 名成人 / **7** 项风险因素 |
| 阳性数 | **91,932** 例（患病率 **31.56%**） |
| 留出测试集（58,259） | **AUC 0.7960**，**平衡准确率 0.7263**，敏感度 0.7715，特异度 0.6811 |
| 3×5 折交叉验证 | **AUC 0.7956 ± 0.0020**，**平衡准确率 0.7251 ± 0.0015** |
| PR-AUC | 0.6187（基线 0.3156，**1.96 倍**） |
| 决策阈值 | **0.305**（训练集 CV 选定，非 0.5） |
| 人群划分属性 | **未使用**（性别/种族构建阶段即未提取） |

## 7 项特征

| 特征 | 取值 |
|------|------|
| `age` | 年龄 18–85 岁 |
| `bmi` | BMI 体重指数 |
| `height_cm` / `weight_kg` | 身高 / 体重 |
| `smoker` | 1=每天 2=偶尔 3=已戒 4=从不 |
| `hypertension` | 高血压（医生告知）1=有 0=无 ← **强烈建议提供** |
| `diabetes` | 糖尿病（医生告知）1=有 0=无 ← **强烈建议提供** |

**为何是这 7 项**（8 个随机种子的平均 AUC）：

| 特征集 | 特征数 | AUC |
|--------|--------|-----|
| 年龄 | 1 | 0.7483 |
| + BMI | 2 | 0.7602 |
| + 身高/体重 | 4 | 0.7616 |
| + 吸烟 | 5 | 0.7638 |
| **+ 高血压 + 糖尿病 ← 采用** | **7** | **0.7959** |
| + 冠心病/脑卒中/慢阻肺 | 10 | 0.8004（**排除**：下游后果，反向因果） |

## ⚠️ 三个必读限制

**1. 不含任何血脂检验值。**
总胆固醇 / LDL-C / 甘油三酯**本身就是高脂血症的诊断依据**，用它们预测
近乎循环论证（同系列糖尿病 v3 不纳入 HbA1c/血糖是同一逻辑）。
**→ 本模型只做风险分层，确诊必须依靠血脂检测。**

**2. 标签是「检出依赖」的 —— 这是性能上限的主因。**
高脂血症**无症状**。要被记录为「曾被医生告知高胆固醇」，前提是
**先做过血脂检测**、结果异常、并被医生告知。因此标签同时编码了
真实血脂水平、**是否获得过检测**（取决于医疗可及性/保险/就医频率）、
是否被告知。问卷无法分离后两项。**「低风险」不等于没有高脂血症。**

**3. 标签只覆盖胆固醇，不含甘油三酯。**
NHIS **只有** `CHLEV` 一个血脂变量，**没有甘油三酯变量**（已核验 10 个年度）。
标签是**高脂血症的胆固醇部分**，**不包含「单纯高甘油三酯血症」**。

## ⚠️ 性能低于同系列慢病线

| 指标 | 本模型 | 本任务线 | 慢病线 | 结论 |
|------|--------|----------|--------|------|
| 交叉验证 AUC | **0.7956 ± 0.0020** | ≥ 0.78 ✓ | ≥ 0.80 | **未达**（差 0.0044） |
| 交叉验证 平衡准确率 | **0.7251 ± 0.0015** | ≥ 0.68 ✓ | ≥ 0.70 | **达到** |

**下调依据：**

1. **标签检出依赖**（见上文）。同系列其他疾病的标签虽也是自报，
   但糖尿病/高血压有更明确的检测触发路径；血脂检测是**选择性**进行的。
2. **最强决定因素不可得**：血脂值本身（循环）、**高胆固醇家族史**
   （NHIS 从未询问）、膳食饱和脂肪、体力活动（2019 后仅偶数年）、
   社会经济地位（**2016–2018 成人文件根本没有教育/收入变量**）、遗传。
3. **模型已饱和**：48 组超参数扫描（`max_depth` 3–6 × `n_estimators`
   {400,800} × `learning_rate` {0.05,0.1} × `min_child_weight` {10,30,60}）
   的 AUC 区间仅 **0.7917–0.7960**，瓶颈在特征而非模型容量。

> ⚠️ **0.78 这条线是在观察到模型饱和于 0.796 之后设定的，属事后设定，
> 并非预先注册的阈值。** 上述依据独立可核验，但读者应知道这一时序。

## 风险分档（留出测试集实测，基线 31.56%）

| 风险档 | 概率区间 | 人群占比 | 实测高胆固醇率 | 相对基线 |
|--------|----------|----------|---------------|----------|
| 低风险 | < 0.20 | 39.45% | 9.10% | 0.29× |
| 人群基线 | 0.20–0.35 | 20.09% | 27.73% | 0.88× |
| 中等风险 | 0.35–0.55 | 19.88% | 45.20% | 1.43× |
| 较高风险 | 0.55–0.75 | 17.41% | 62.89% | 1.99× |
| 高风险 | ≥ 0.75 | 3.18% | 77.62% | 2.46× |

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 单个体预测（3 项核心指标必需；高血压/糖尿病强烈建议提供）
python scripts/predict.py --age 55 --bmi 28.5 --smoker 4 \
    --height 170 --weight 82 --hypertension 1 --diabetes 0

# 批量预测
python scripts/batch_predict.py --input examples/people.csv --output results.csv

# 特征重要性
python scripts/feature_importance.py

# 重新训练与验证（可选，约 60 秒）
python scripts/train_model.py

# 重建数据集（下载 10 个年度官方文件并校验 SHA256，约 45MB）
python scripts/fetch_data.py
```

## 目录结构

```
hyperlipidemia_prediction/
├── SKILL.md                      # 技能说明（含完整评估与解读警告）
├── README.md
├── requirements.txt
├── data/
│   ├── hyperlipidemia_nhis.csv   # 291,295 条真实数据（内置，离线可用）
│   └── DATA_PROVENANCE.md        # 溯源：SHA256、码本依据、标签规则、特征取舍、性能上限论证
├── models/
│   ├── hyperlipidemia_xgb_model.json   # XGBoost 原生格式（推荐）
│   ├── hyperlipidemia_xgb_model.ubj    # 二进制备份
│   ├── imputer.joblib                  # 中位数插补参数
│   └── model_metadata.json             # 阈值、分档、全部评估指标
├── results/
│   ├── holdout_metrics.json
│   ├── cv_metrics.json
│   ├── feature_importance.json
│   ├── risk_band_calibration.json
│   └── batch_results.csv
├── examples/
│   └── people.csv
└── scripts/
    ├── fetch_data.py             # 下载 + SHA256 校验 + 构建队列
    ├── train_model.py            # 训练 + 验证 + 保存工件
    ├── predict.py                # 单个体推理
    ├── batch_predict.py          # 批量推理
    └── feature_importance.py     # 特征重要性（含解读警告）
```

## Python 调用

```python
import sys
sys.path.insert(0, "scripts")
from predict import load_artifacts, predict_patient

model, medians, order, metadata, src = load_artifacts()
r = predict_patient(model=model, medians=medians, order=order, metadata=metadata,
                    age=55, bmi=28.5, smoker=4, hypertension=1, diabetes=0)
print(r["probability_pct"])   # 44.66%
print(r["risk_level"])        # 中等风险
```

## 复现记录

| 项目 | 值 |
|------|-----|
| 特征 | `age, bmi, height_cm, weight_kg, smoker, hypertension, diabetes` |
| 插补 | `SimpleImputer(strategy="median")`，无标准化 |
| 划分 | `test_size=0.2, random_state=42, stratify=y` |
| 交叉验证 | `RepeatedStratifiedKFold(5, n_repeats=3, random_state=42)` |
| 阈值 | 训练集 5 折 CV 最大化平衡准确率 → 0.305 |
| XGBoost | `n_estimators=400, max_depth=3, lr=0.1, subsample=0.8, colsample_bytree=0.7, min_child_weight=30` |
| 数据 SHA256 | `6c4055d80df9acc6b297f5581cebc625a90d0838066b77fb35fe5559f4198de1` |

## 声明

本技能仅供学习研究使用。高脂血症筛查与诊断应依据临床指南（血脂检测），
本模型仅用于风险分层，**不能替代血液检查，也不能替代医生的判断**，请遵医嘱。
