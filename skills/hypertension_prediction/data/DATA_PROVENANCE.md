# 数据溯源说明（DATA PROVENANCE）

本 skill 使用 **NHANES 2017-2018（美国国家健康与营养调查）**，
来源权威、可逐字节追溯，且**目标人群为全人群，不按性别或种族划分**。

## 1. 数据来源

| 项目 | 内容 |
|------|------|
| 调查项目 | NHANES (National Health and Nutrition Examination Survey) |
| 执行机构 | U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS) |
| 调查周期 | 2017–2018（cycle J） |
| 目标人群 | **美国非机构化平民人口**（civilian noninstitutionalized population），不限性别、种族、地区、收入 |
| 抽样方式 | 复杂多阶段概率抽样，全国代表性 |
| 官方页面 | https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017 |
| 文件下载基址 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{文件名}.xpt |
| 数据许可 | 美国政府公开数据（Public Domain），可自由使用 |

### 为什么选 NHANES

1. **全人群**：全国代表性抽样，覆盖男女、各年龄段、各种族/族裔、各收入层。
   本建模队列中男性 2,549 人、女性 2,701 人，**未按性别划分人群**。
2. **客观测量**：血压由经过培训的技术人员用标准化流程**实测**（而非自我报告），
   比问卷自报数据更可靠。
3. **可追溯**：原始数据文件由 CDC 官网直接下载，每个文件均有官方公布的内容，
   并可逐文件校验哈希值。

## 2. 使用的原始文件与校验值

| 文件 | 内容 | SHA256 |
|------|------|--------|
| DEMO_J.xpt | 人口学（年龄、性别、种族、收入、教育） | `c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec` |
| BPX_J.xpt | 血压实测（最多 4 次读数） | `f98f1d9a4c18173e715c749b4294f6f1525140d5ad5751316bb17cc038fb3423` |
| BPQ_J.xpt | 血压/胆固醇问卷（含是否服降压药） | `63cfe1c331a1e7d3534328ac86312a21ffee3e01aa189bc1f74d06855239e5aa` |
| BMX_J.xpt | 体格测量（身高、体重、BMI、腰围） | `8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838a` |
| SMQ_J.xpt | 吸烟问卷 | `73c3708e540cbf520a566508aa629c3e4ba53cad77dae9388ca36384646455f4` |
| DIQ_J.xpt | 糖尿病问卷 | `1ecbf5360dfc331d1efbf32198553dc30e9a1f4cff0a907ed30ca72bac797f89` |
| GHB_J.xpt | 糖化血红蛋白 HbA1c | `35f07094573a0061a03ed609a5a363b34eb1b1c7065d1623b43d72e132a8a654` |
| TCHOL_J.xpt | 总胆固醇 | `0291a6a4f6d82dac8392c8c992b519946824dfeda01647935520cf506ed9f4ab` |
| HDL_J.xpt | 高密度脂蛋白 HDL | `9f4eb41b89f0c9f3d6262f873e671eeb2f2eb40581a8426b2a3c8fd4fc19439e` |
| BIOPRO_J.xpt | 生化全套（肌酐、尿酸等） | `5bcd5722c1892883b96a9d7fed0befadabacae313f800fadc0133cd5dd00c4c6` |

上述 SHA256 已内置于 `scripts/fetch_data.py`，下载时**自动校验**，任一文件不符即中止。

| 派生文件 | SHA256 |
|----------|--------|
| `data/hypertension_nhanes.csv` | `e08ec328a3cd6cec11b8beee90352753cc3db887857944c14a591252e1b2be5f` |

已实测：删除内置 CSV 后重新执行 `python scripts/fetch_data.py`，
从 CDC 重新下载全部 10 个文件并重建，产出的 CSV 与内置文件**逐字节一致**。

## 3. 建模队列构建规则（完全可复现）

1. **合并**：按 `SEQN` 左连接上述 10 个文件。
2. **血压取值**：取第 2~4 次读数的均值（NHANES 标准做法，降低单次测量噪声）；
   若第 2~4 次全缺失，回退到第 1 次读数。
3. **高血压判定（ACC/AHA 2017 标准，标签）**：
   平均收缩压 SBP ≥ 130 mmHg **或** 平均舒张压 DBP ≥ 80 mmHg **或**
   正在服用降压药（`BPQ050A = 1`）。
4. **纳入标准**：年龄 ≥ 18 岁，且收缩压/舒张压均有有效测量值。
5. **结果**：5,250 人，阳性（高血压）2,857 人 / 阴性 2,393 人，患病率 **54.4%**。

### 特征列表（14 个，均为常规体检/问卷/血检指标）

| 列名 | 含义 | 单位 | 来源文件 |
|------|------|------|----------|
| age | 年龄 | 岁 | DEMO_J |
| bmi | BMI 体重指数 | kg/m² | BMX_J |
| waist_cm | 腰围 | cm | BMX_J |
| height_cm | 身高 | cm | BMX_J |
| education | 教育程度（1=低于高中 2=高中 3=大专 4=本科 5=研究生） | 编码 | DEMO_J |
| income_pir | 家庭收入贫困比 | 比值 | DEMO_J |
| smoker | 是否吸烟（一生 ≥100 支） | 1/0 | SMQ_J |
| diabetes | 是否患糖尿病（自报，含"临界"归为否） | 1/0 | DIQ_J |
| pulse | 静息脉搏 | 次/分 | BPX_J |
| hba1c | 糖化血红蛋白 | % | GHB_J |
| cholesterol | 总胆固醇 | mg/dL | TCHOL_J |
| hdl | 高密度脂蛋白 | mg/dL | HDL_J |
| creatinine | 血清肌酐 | mg/dL | BIOPRO_J |
| uric_acid | 血清尿酸 | mg/dL | BIOPRO_J |

> **重要**：性别（`RIAGENDR`）与种族（`RIDRETH3`）**未纳入模型特征**。
> 理由是实测二者贡献很小 —— 纳入后 10×5 折交叉验证 AUC 从 0.8234 提升到 0.8301
> （+0.007），准确率基本持平（0.7597 → 0.7594）。因此默认模型不使用任何
> 人群划分属性，适用范围更广。
> 如确需纳入，可运行 `python scripts/train_model.py --include-sensitive` 重新训练。

### 被排除的字段（避免标签泄漏）

`BPQ020`（是否被告知有高血压）、`BPQ030`、`BPQ040A`（是否因高血压服药）、
`BPQ050A`（现在是否服降压药）以及 `SBP`/`DBP` 本身均为标签定义的一部分，
**未作为特征使用**。

## 4. 缺失值

派生 CSV 直接保留原始缺失（空值），由训练/推理时的
`SimpleImputer(strategy="median")` 按训练中位数插补。各列缺失数：

| 列 | 缺失 | 列 | 缺失 |
|----|------|----|------|
| age | 0 | hba1c | 241 |
| bmi | 71 | cholesterol | 323 |
| waist_cm | 252 | hdl | 323 |
| height_cm | 63 | creatinine | 343 |
| education | 259 | uric_acid | 345 |
| income_pir | 671 | smoker | 0 |
| diabetes | 4 | pulse | 0 |

插补中位数完整记录在 `models/model_metadata.json` 的
`preprocessing.imputer_statistics`。

## 5. 复现方式

```bash
# 从头重建数据集（自动下载 + 逐文件 SHA256 校验 + 重建队列）
python scripts/fetch_data.py

# 仅校验已有 CSV
python scripts/fetch_data.py --check

# 重新训练并验证
python scripts/train_model.py
```

## 6. 使用限制与声明

1. **人群适用性**：模型基于**美国**非机构化平民人口的调查数据（2017–2018），
   应用于其他国家和地区人群时需谨慎。
2. **年龄段**：队列仅含 ≥18 岁成人，不适用于儿童青少年。
3. **标签口径**：高血压判定采用 ACC/AHA 2017 的 130/80 mmHg 标准。
   若采用其他标准（如 JNC7 的 140/90），患病率与模型表现会不同。
4. **单次访视测量**：血压取自单次访视的多次读数均值，仍存在测量波动，
   这是本任务精度的主要限制来源之一。
5. **本研究未使用抽样权重**：训练时未使用 NHANES 的复杂抽样权重，
   因此模型输出**不可解释为全国患病率估计**，仅用于个体风险分层。
6. **临床用途**：本模型仅用于健康风险筛查的辅助参考与教学演示，
   **不能替代专业医生的诊断**。
