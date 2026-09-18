# 数据说明与医疗免责声明 / Data Notice and Medical Disclaimer

本文件是 `LICENSE`（MIT）之外**独立的**补充说明。
/LICENSE 仅包含 MIT 许可证正文，以便 GitHub 正确识别许可证类型。

---

## 数据说明 / Data Notice

本项目使用的数据集均为**美国政府公共领域数据**，来源为：

- 美国疾病控制与预防中心 (CDC)
- 美国国家卫生统计中心 (NCHS)

**本仓库不再分发原始数据。** 各 skill 的 `scripts/fetch_data.py` 会从官方
端点下载并通过 SHA256 校验；来源 URL、许可证与校验值记录在
各 skill 的 `data/DATA_PROVENANCE.md`。

数据本身不受本项目 MIT 许可证约束，其使用请遵守各自提供方的条款。

The datasets used by this project are U.S. government public-domain data
sourced from the Centers for Disease Control and Prevention (CDC) and the
National Center for Health Statistics (NCHS). This repository does not
redistribute raw source data; download scripts retrieve it from official
endpoints and verify it against SHA256 checksums. The data itself is not
covered by this project's MIT license; please follow each provider's terms.

---

## 医疗免责声明 / Medical Disclaimer

**本软件仅供研究与教学使用。它不是医疗器械，不得用于临床诊断、治疗或任何
医疗决策。** 所有模型反映的都是流行病学数据上的统计关联，个体风险判断
请务必咨询执业医师。

This software is provided for research and educational purposes only. It is
not a medical device and must not be used for clinical diagnosis, treatment,
or any medical decision-making. All models reflect statistical associations
in epidemiological data only. Always consult a qualified healthcare
professional for individual risk assessment.
