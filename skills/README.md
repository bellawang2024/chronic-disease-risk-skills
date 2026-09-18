# 慢病预测模型索引 / Chronic Disease Risk Models

本目录收录 7 个基于**美国全国性健康调查**构建的慢病风险预测模型。
全部模型遵循同一套方法论约定，因此结果之间可以横向比较。

共 **7** 个模型包，CV AUC 区间 **0.7956 – 0.8460**。

## 总览

| Skill | 疾病 | 数据集 | 样本量 | 阳性率 | 特征数 | CV AUC | CV 平衡准确率 | 决策阈值 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| [`chd_prediction`](./chd_prediction/) | 冠心病 / 心肌梗死 | BRFSS 2017 | 445,872 | 8.83% | 15 | 0.8460 | 0.7684 | 0.08 |
| [`copd_prediction`](./copd_prediction/) | 慢性阻塞性肺疾病 (COPD) | BRFSS 2017 | 447,718 | 8.39% | 14 | 0.8415 | 0.7641 | 0.09 |
| [`stroke_prediction`](./stroke_prediction/) | 脑卒中 | BRFSS 2017 | 448,666 | 4.22% | 15 | 0.8149 | 0.7436 | 0.04 |
| [`hypertension_prediction`](./hypertension_prediction/) | 高血压 | NHANES 2017-2018 | 5,250 | 54.42% | 14 | 0.8234 | — | 0.5 |
| [`diabetes_prediction_v3`](./diabetes_prediction_v3/) | 糖尿病 (v3) | NHIS 2016-2025 | 291,945 | 10.88% | 7 | 0.8124 | 0.7419 | 0.105 |
| [`hyperlipidemia_prediction`](./hyperlipidemia_prediction/) | 高脂血症 | NHIS 2016-2025 | 291,295 | 31.56% | 7 | 0.7956 | 0.7251 | 0.305 |
| [`diabetes_prediction`](./diabetes_prediction/) | 糖尿病 (v2 · Pima) | Pima Indians Diabetes Database | 768 | 34.90% | 8 | 0.8379 | — | 0.5 |

> **所有数字均可复现**：上表由 `tools/build_skill_index.py` 直接从各 skill 的
> `models/model_metadata.json` 与 `results/*.json` 读取生成，无任何手工转录。
> 可自行运行 `python3 tools/build_skill_index.py --check` 校验表格与模型产物是否一致。

> 表中为 **交叉验证 (CV)** 指标（5 折 × 3 次重复 = 15 折）；
> 留出集指标见各 skill 的 `results/holdout_metrics.json`，两者通常相差 ±0.01 以内。

> **为什么不用准确率 (Accuracy)**：本组任务阳性率最低仅 4.22%，
> 「全部预测为阴性」这一无脑策略就能拿到 **95.78%** 的准确率。
> 因此一律以 **AUC + 平衡准确率 + 敏感度/特异度** 为准，
> 常规准确率仅在 `results/holdout_metrics.json` 里作为反面参照保留。

## 使用方式

每个目录都是一个自包含的 skill 包，结构与用法完全一致：

```
<skill_name>/
├── SKILL.md              # 技能说明（frontmatter + 完整方法论）
├── README.md             # 快速上手
├── scripts/
│   ├── fetch_data.py     # 按 SHA256 校验下载原始数据
│   ├── train_model.py    # 完整训练流程（可复现）
│   ├── predict.py        # 单条/小批量推理
│   ├── batch_predict.py  # 批量推理（CSV 进 CSV 出）
│   └── feature_importance.py
├── models/               # 已训练模型（推理开箱即用，无需下载数据）
│   ├── *_xgb_model.json  # XGBoost 原生格式（推荐）
│   ├── *_xgb_model.ubj   # 二进制格式（体积更小）
│   ├── imputer.joblib    # 中位数插补器
│   └── model_metadata.json
├── results/              # 训练产出的全部指标
├── examples/             # 示例输入
└── data/DATA_PROVENANCE.md  # 数据来源、许可、SHA256
```

推理只需 `xgboost` + `numpy`；**不下载数据也能直接跑推理**，
因为 `models/` 已随仓库提交。只有重新训练才需要执行 `fetch_data.py`。

```bash
pip install -r chd_prediction/requirements.txt
python chd_prediction/scripts/predict.py --help
```

## 逐个说明

### `chd_prediction` — 冠心病 / 心肌梗死

**数据集**：BRFSS 2017 (Behavioral Risk Factor Surveillance System)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC)  
**年份**：2017  
**样本量**：445,872（阳性 39,357，阳性率 8.83%）  
**特征数**：15 — `age`, `bmi`, `height_cm`, `weight_kg`, `smoker`, `diabetes`, `hypertension`, `high_chol`, `gen_health`, `exercise`, `education`, `income`, `drinks_wk`, `depression`, `copd`  
**预测目标**：BRFSS 派生变量 _MICHD = 曾被医生告知患有冠心病(CHD)或心肌梗死(MI)  
**CV 表现**：AUC 0.8460、平衡准确率 0.7684、敏感度 0.8120、特异度 0.7249、PR-AUC 0.3513（5 折 × 3 次）  
**留出集**：AUC 0.8479、平衡准确率 0.7693、PR-AUC 0.3537（基线 0.0883）  
**决策阈值**：`0.08`  
**数据 SHA256**：`a43af90ee0f846f6e0ba72e47c1068a2a6deaf9089e406c0359e2af6bb6c8055`  

特征中**不含** SEX、RACE —— 派生数据集在提取阶段就没有取这两列，模型结构上无法使用人群划分属性。

### `copd_prediction` — 慢性阻塞性肺疾病 (COPD)

**数据集**：BRFSS 2017 (Behavioral Risk Factor Surveillance System)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC)  
**年份**：2017  
**样本量**：447,718（阳性 37,577，阳性率 8.39%）  
**特征数**：14 — `age`, `bmi`, `height_cm`, `weight_kg`, `smoker`, `diabetes`, `hypertension`, `high_chol`, `gen_health`, `exercise`, `education`, `income`, `drinks_wk`, `depression`  
**预测目标**：BRFSS 变量 CHCCOPD1 = 曾被医生告知患有慢阻肺(COPD)、肺气肿或慢性支气管炎（1=是 2=否）  
**CV 表现**：AUC 0.8415、平衡准确率 0.7641、敏感度 0.7630、特异度 0.7653、PR-AUC 0.3672（5 折 × 3 次）  
**留出集**：AUC 0.8389、平衡准确率 0.7609、PR-AUC 0.3614（基线 0.0839）  
**决策阈值**：`0.09`  
**数据 SHA256**：`386ab988726a6ad6047bbd0d8858058293a6ed5b546ff70430b87fbf275c65d7`  

目标为自报「曾被医生告知患有 COPD / 肺气肿 / 慢性支气管炎」。

### `stroke_prediction` — 脑卒中

**数据集**：BRFSS 2017 (Behavioral Risk Factor Surveillance System)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC)  
**年份**：2017  
**样本量**：448,666（阳性 18,956，阳性率 4.22%）  
**特征数**：15 — `age`, `bmi`, `height_cm`, `weight_kg`, `smoker`, `diabetes`, `hypertension`, `high_chol`, `gen_health`, `exercise`, `education`, `income`, `drinks_wk`, `depression`, `copd`  
**预测目标**：BRFSS 变量 CVDSTRK3 = 曾被医生告知患有脑卒中（1=是 2=否）  
**CV 表现**：AUC 0.8149、平衡准确率 0.7436、敏感度 0.8058、特异度 0.6814、PR-AUC 0.1566（5 折 × 3 次）  
**留出集**：AUC 0.8187、平衡准确率 0.7478、PR-AUC 0.1607（基线 0.0422）  
**决策阈值**：`0.04`  
**数据 SHA256**：`360136a4ea1b6888315c671975e38b6ad5053f2afdb74cbf824f26239158b238`  

阳性率仅 4.22%，阈值 0.04 —— 该阈值由训练集交叉验证选出，**不是** 0.5；固定 0.5 会让敏感度塌掉。

### `hypertension_prediction` — 高血压

**数据集**：NHANES 2017-2018 (National Health and Nutrition Examination Survey)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)  
**样本量**：5,250（阳性 2,857，阳性率 54.42%）  
**特征数**：14 — `age`, `bmi`, `waist_cm`, `height_cm`, `education`, `income_pir`, `smoker`, `diabetes`, `pulse`, `hba1c`, `cholesterol`, `hdl`, `creatinine`, `uric_acid`  
**预测目标**：ACC/AHA 2017: 平均 SBP>=130 或 平均 DBP>=80 或 正在服用降压药 (BPQ050A=1)  
**CV 表现**：AUC 0.8234、平衡准确率 —、敏感度 0.8338、特异度 —（5 折 × 10 次）  
**留出集**：AUC 0.8209、平衡准确率 —  
**决策阈值**：`0.5`  
**数据 SHA256**：`e08ec328a3cd6cec11b8beee90352753cc3db887857944c14a591252e1b2be5f`  

⚠️ **早期版本**：使用 NHANES 生物标志物（HbA1c、血清尿酸、腰围等）作为特征，与后续「只用可自报的问卷变量」的原则不同，**指标不可与 v3 系列直接横向比较**。本 skill 产出于早期格式，交叉验证仅记录了 AUC / 准确率 / 精确率 / 召回率 / F1，未记录平衡准确率，故上表该列为 `—`。

### `diabetes_prediction_v3` — 糖尿病 (v3)

**数据集**：NHIS 2016-2025 (National Health Interview Survey, Sample Adult)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)  
**年份**：[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]  
**样本量**：291,945（阳性 31,760，阳性率 10.88%）  
**特征数**：7 — `age`, `bmi`, `height_cm`, `weight_kg`, `smoker`, `hypertension`, `high_chol`  
**预测目标**：2018 及以前用 DIBEV1；2019 及以后用 DIBEV_A。1=是 → 阳性；2=否 / 3=临界(糖尿病前期) → 阴性；7/9 → 缺失  
**CV 表现**：AUC 0.8124、平衡准确率 0.7419、敏感度 0.7917、特异度 0.6922、PR-AUC 0.3255（5 折 × 3 次）  
**留出集**：AUC 0.8132、平衡准确率 0.7442、PR-AUC 0.3310（基线 0.1088）  
**决策阈值**：`0.105`  
**数据 SHA256**：`ca891b65297a6841f1d099a41b87e11d17d57ea09dbe2c8a30481454708af213`  

当前推荐版本：NHIS 全人群、7 个纯问卷特征、无性别/种族划分。

### `hyperlipidemia_prediction` — 高脂血症

**数据集**：NHIS 2016-2025 (National Health Interview Survey, Sample Adult)  
**提供方**：U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)  
**年份**：[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]  
**样本量**：291,295（阳性 91,932，阳性率 31.56%）  
**特征数**：7 — `age`, `bmi`, `height_cm`, `weight_kg`, `smoker`, `hypertension`, `diabetes`  
**预测目标**：2018 及以前用 CHLEV；2019 及以后用 CHLEV_A。1=是 → 阳性；2=否 → 阴性；7/8/9（拒答/未查明/不知道）→ 缺失  
**CV 表现**：AUC 0.7956、平衡准确率 0.7251、敏感度 0.7514、特异度 0.6988、PR-AUC 0.6185（5 折 × 3 次）  
**留出集**：AUC 0.7960、平衡准确率 0.7263、PR-AUC 0.6187（基线 0.3156）  
**决策阈值**：`0.305`  
**数据 SHA256**：`6c4055d80df9acc6b297f5581cebc625a90d0838066b77fb35fe5559f4198de1`  

接受线 AUC 0.78 / 平衡准确率 0.68 是**事后设定、未预注册**；超参搜索 48 组配置 AUC 区间 [0.7917, 0.796]，已接近该特征集上限。

### `diabetes_prediction` — 糖尿病 (v2 · Pima)

**数据集**：Pima Indians Diabetes Database  
**提供方**：National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)  
**样本量**：768（阳性 268，阳性率 34.90%）  
**特征数**：8 — `preg`, `plas`, `pres`, `skin`, `insu`, `mass`, `pedi`, `age`  
**预测目标**：label  
**CV 表现**：AUC 0.8379、平衡准确率 —、敏感度 0.6099、特异度 —（5 折 × 10 次）  
**留出集**：AUC 0.8257、平衡准确率 —  
**决策阈值**：`0.5`  
**数据 SHA256**：`fa9f4490a5e0d9eb7a0a65e758388737ac74f3828d4c877acc6145abcbbb2997`  

⚠️ **历史版本，不建议新项目使用**。Pima 数据集有三重人群限制：仅女性、仅皮马印第安人、且为 1990 年代队列。保留仅为记录演进过程。早期格式未记录平衡准确率，故上表该列为 `—`。

---

## 统一的方法论约定

这组模型共享同一套流程约定，这也是它们可以横向比较的前提：

1. **只用可自报/可获取的变量**，避免把诊断标准本身当特征（例如用「是否服用降压药」预测高血压，属于标签泄漏）。
2. **不做性别、种族等子集切分**，除非该结局在数据源中只对特定人群询问（乳腺癌是唯一例外，已在表中标注）。
3. **中位数插补**，不做标准化 —— XGBoost 对特征尺度不敏感。
4. **阈值在训练集交叉验证上选取**，最大化平衡准确率，绝不使用测试集；
   罕见结局的阈值通常远低于 0.5。
5. **分层划分** `train_test_split(test_size=0.2, stratify=y, random_state=42)`。
6. **报告 AUC + PR-AUC + 平衡准确率 + 敏感度/特异度**，不报告裸准确率。
7. **模型携带真实特征名**（自定义 `NamedXGB` 子类写入 `feature_names`），不会出现 `f0, f1, ...` 这类无意义名称。

## 数据未随仓库提交

原始 CSV 体积较大（本仓库合计 约 69 MB），未纳入 git。
每个 skill 的 `scripts/fetch_data.py` 会**按 SHA256 校验**下载，
`data/DATA_PROVENANCE.md` 记录了来源 URL、许可证与校验值。
这样既保持仓库轻量，又保证可复现性 —— 校验和比直接提交文件更能证明「拿到的确实是同一份数据」。

## 免责声明

**本项目仅用于研究与教学，不构成医疗建议，不得用于临床诊断。**
所有模型均为流行病学数据上的统计关联，个体风险判断请咨询执业医师。
数据版权归各自提供方所有，使用请遵守原始许可。
