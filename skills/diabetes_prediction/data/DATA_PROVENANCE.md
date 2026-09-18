# 数据溯源说明（DATA PROVENANCE）

本 skill 使用 **Pima Indians Diabetes Database**，来源权威、可逐字节追溯。

## 1. 数据来源链条

| 环节 | 内容 |
|------|------|
| 原始所有者 | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA |
| 数据捐赠者 | Vincent Sigillito，Johns Hopkins University, Applied Physics Laboratory |
| 数据接收日期 | 1990-05-09 |
| 分发渠道（本次使用） | OpenML，dataset id **37**，licence: Public |
| OpenML 下载地址 | https://openml.org/data/v1/download/37/diabetes.arff |
| OpenML 数据集页面 | https://www.openml.org/d/37 |
| 原始 UCI 页面 | https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes |
| 引用文献 | Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proceedings of the Symposium on Computer Applications in Medical Care, 261–265. |

## 2. 校验值（可用于验证数据未被篡改）

| 文件 | 校验值 |
|------|--------|
| `diabetes.arff`（OpenML 原始文件） | MD5 `3cbaa3e54586aa88cf6aacb4033e4470` |
| `diabetes.arff`（OpenML 原始文件） | SHA256 `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` |
| `data/pima_indians_diabetes.csv`（本 skill 内置） | SHA256 `fa9f4490a5e0d9eb7a0a65e758388737ac74f3828d4c877acc6145abcbbb2997` |

MD5 `3cbaa3e54586aa88cf6aacb4033e4470` 与 OpenML 官方 API 公布的 `md5_checksum` 字段**完全一致**
（见 https://www.openml.org/api/v1/json/data/37 ）。

## 3. 独立交叉验证

为避免单一来源的风险，本次构建还从**独立镜像**
（`https://cdn.jsdelivr.net/gh/jbrownlee/Datasets@master/pima-indians-diabetes.data.csv`）
下载了同一数据集，逐值比对结果：**768 条记录、9 列，全部数值完全一致**。

## 4. 数据规格

| 项目 | 值 |
|------|-----|
| 样本数 | 768 |
| 特征数 | 8（数值型）+ 1 个类别标签 |
| 类别分布 | 阳性（糖尿病）268 / 阴性 500 → 阳性率 34.9% |
| 缺失值 | 官方标注为 None，但 5 个列中以 0 表示实际缺失（见下） |
| 人群限制 | 全部为 **≥21 岁的 Pima 印第安女性**（数据集原始纳入标准） |

### 特征列表

| 序号 | 列名 | 含义 | 单位 |
|------|------|------|------|
| 1 | preg | 怀孕次数 | 次 |
| 2 | plas | 口服糖耐量试验 2 小时血浆葡萄糖浓度 | mg/dL |
| 3 | pres | 舒张压 | mm Hg |
| 4 | skin | 三头肌皮褶厚度 | mm |
| 5 | insu | 2 小时血清胰岛素 | mu U/ml |
| 6 | mass | 体重指数 BMI | kg/m² |
| 7 | pedi | 糖尿病家族遗传函数 | — |
| 8 | age | 年龄 | 岁 |
| 9 | label | 类别：1 = 按 WHO 标准判定为糖尿病，0 = 非糖尿病 | — |

### 关于"0 即缺失"

以下 5 个列在生理上不可能为 0，数据集用 0 表示缺失。训练与推理时统一按缺失处理，
用中位数插补（插补中位数记录在 `models/model_metadata.json` 的
`preprocessing.imputer_statistics`）：

| 列 | 0 的个数 | 占比 |
|----|---------|------|
| plas | 5 | 0.7% |
| pres | 35 | 4.6% |
| skin | 227 | 29.6% |
| insu | 374 | 48.7% |
| mass | 11 | 1.4% |

## 5. 复现方式

```bash
# 重新下载并校验（无需人工干预，脚本内置 MD5 校验）
python scripts/fetch_data.py

# 仅校验已有的内置数据
python scripts/fetch_data.py --check
```

## 6. 使用限制与声明

1. **人群限制**：数据仅涵盖 ≥21 岁的 Pima 印第安女性，模型结论**不可直接外推**到
   男性、儿童、其他族裔或其他地区人群。
2. **数据年代**：数据采集于 1990 年前后，反映的是当时的临床测量与诊断标准。
3. **许可**：OpenML 标注 licence 为 **Public**；UCI/OpenML 均提供公开下载。
   使用时请引用上方 Smith et al. (1988) 文献。
4. **临床用途**：本模型仅用于健康风险筛查的辅助参考与教学演示，**不能替代专业医生的诊断**。
