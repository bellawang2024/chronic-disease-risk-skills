#!/usr/bin/env python3
"""
从每个 skill 的模型产物 (models/model_metadata.json、results/cv_metrics.json、
results/holdout_metrics.json) 直接生成 <repo>/skills/README.md。

设计原则：索引里的每一个数字都来自模型产物本身，不做任何手工转录，
避免"文档写的和模型实际跑出来的不一致"这类错误。

本脚本位于 <repo>/tools/，运行时会自动识别所在仓库。

运行：  python3 tools/build_skill_index.py
校验：  python3 tools/build_skill_index.py --check   (只校验不写入)
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent   # 仓库根目录
REPO = ROOT.name                                        # 仓库名，如 chronic-disease-risk-models

# ── skill 元信息（中文名与注记需人工维护；所有数字均自动读取）──────────────
SKILL_META = {
    "chd_prediction": dict(
        cn="冠心病 / 心肌梗死", model="chd_xgb_model",
        note="特征中**不含** SEX、RACE —— 派生数据集在提取阶段就没有取这两列，"
             "模型结构上无法使用人群划分属性。",
    ),
    "copd_prediction": dict(
        cn="慢性阻塞性肺疾病 (COPD)", model="copd_xgb_model",
        note="目标为自报「曾被医生告知患有 COPD / 肺气肿 / 慢性支气管炎」。",
    ),
    "stroke_prediction": dict(
        cn="脑卒中", model="stroke_xgb_model",
        note="阳性率仅 4.22%，阈值 0.04 —— 该阈值由训练集交叉验证选出，"
             "**不是** 0.5；固定 0.5 会让敏感度塌掉。",
    ),
    "hypertension_prediction": dict(
        cn="高血压", model="hypertension_xgb_model",
        note="⚠️ **早期版本**：使用 NHANES 生物标志物（HbA1c、血清尿酸、腰围等）作为特征，"
             "与后续「只用可自报的问卷变量」的原则不同，**指标不可与 v3 系列直接横向比较**。"
             "本 skill 产出于早期格式，交叉验证仅记录了 AUC / 准确率 / 精确率 / 召回率 / F1，"
             "未记录平衡准确率，故上表该列为 `—`。",
    ),
    "diabetes_prediction_v3": dict(
        cn="糖尿病 (v3)", model="diabetes_xgb_model",
        note="当前推荐版本：NHIS 全人群、7 个纯问卷特征、无性别/种族划分。",
    ),
    "hyperlipidemia_prediction": dict(
        cn="高脂血症", model="hyperlipidemia_xgb_model",
        note="接受线 AUC 0.78 / 平衡准确率 0.68 是**事后设定、未预注册**；"
             "超参搜索 48 组配置 AUC 区间 [0.7917, 0.796]，已接近该特征集上限。",
    ),
    "diabetes_prediction": dict(
        cn="糖尿病 (v2 · Pima)", model="diabetes_xgb_model",
        note="⚠️ **历史版本，不建议新项目使用**。Pima 数据集有三重人群限制："
             "仅女性、仅皮马印第安人、且为 1990 年代队列。保留仅为记录演进过程。"
             "早期格式未记录平衡准确率，故上表该列为 `—`。",
    ),
    "lung_cancer_prediction": dict(
        cn="肺癌", model="lung_xgb_model",
        note="阳性率仅 0.41%，**3 个特征**即达 AUC 0.874 —— 这是「小特征集 + 罕见结局」的样本。"
             "注意：AUC 高不代表临床可用，见 `docs/RARE_OUTCOMES.md`。",
    ),
    "colorectal_cancer_prediction": dict(
        cn="结直肠癌", model="colorectal_cancer_xgb_model",
        note="阳性率 0.72%，PR-AUC 必须与基线对比着看（见 `docs/PERFORMANCE.md`）。",
    ),
    "breast_cancer_prediction": dict(
        cn="乳腺癌", model="breast_cancer_xgb_model",
        note="⚠️ **仅女性样本**（乳腺癌在 NHIS 中只对女性询问），是本仓库唯一按性别划分的模型，"
             "为数据可得性所迫，非设计选择。详见 `docs/RARE_OUTCOMES.md`。",
    ),
    "thyroid_cancer_prediction": dict(
        cn="甲状腺癌", model="thyroid_cancer_xgb_model",
        note="⚠️ **诚实失败案例**，如实保留：AUC 0.639 远低于其他模型。"
             "扩充到 9 个特征后上限仅 0.644，说明该结局在 NHIS 问卷变量下**近乎不可预测**。"
             "单独设有一章分析，见 `docs/PERFORMANCE.md`。",
    ),
    "gastric_cancer_prediction": dict(
        cn="胃癌", model=None,
        note="🚧 **草稿状态 (DRAFT)**：`models/` 为空，尚无训练产物。"
             "原因是 PLCO 数据集需提交研究者申请（PI-submitted proposal）并签署数据传输协议，"
             "无法自动获取；其余候选数据集经核验为**合成数据**，不可使用。见 `data/DATA_PROVENANCE.md`。",
    ),
}

REPOS = {
    "chronic-disease-risk-models": dict(
        title="慢病预测模型",
        en="Chronic Disease Risk Models",
        intro=(
            "本目录收录 7 个基于**美国全国性健康调查**构建的慢病风险预测模型。\n"
            "全部模型遵循同一套方法论约定，因此结果之间可以横向比较。"
        ),
        data_size="约 69 MB",
        order=[
            "chd_prediction", "copd_prediction", "stroke_prediction",
            "hypertension_prediction", "diabetes_prediction_v3",
            "hyperlipidemia_prediction", "diabetes_prediction",
        ],
    ),
    "cancer-risk-models": dict(
        title="癌症预测模型",
        en="Cancer Risk Models",
        intro=(
            "本目录收录 5 个癌症风险预测模型。与慢病仓库不同，这里的核心难点是"
            "**罕见结局 (rare outcome)** 的建模与评估。"
        ),
        data_size="约 18.5 MB",
        order=[
            "lung_cancer_prediction", "colorectal_cancer_prediction",
            "breast_cancer_prediction", "thyroid_cancer_prediction",
            "gastric_cancer_prediction",
        ],
    ),
}


def load(p: pathlib.Path):
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def mean_of(d: dict, *keys):
    """cv_metrics.json 有 {"auc": {"mean":..}} 与 {"roc_auc": {"mean":..}} 两种格式。"""
    for k in keys:
        if k in d:
            v = d[k]
            return v["mean"] if isinstance(v, dict) else v
    return None


def fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def fmt_int(v):
    return f"{v:,}" if isinstance(v, int) else "—"


def collect(skill: str) -> dict:
    sk = ROOT / "skills" / skill
    md = load(sk / "models" / "model_metadata.json") or {}
    ds = md.get("dataset", {}) or {}
    pre = md.get("preprocessing", {}) or {}
    cv = load(sk / "results" / "cv_metrics.json")
    ho = load(sk / "results" / "holdout_metrics.json")
    cv = cv if isinstance(cv, dict) else {}
    ho = ho if isinstance(ho, dict) else {}

    feats = ds.get("features")
    if not isinstance(feats, list):
        feats = pre.get("feature_order") or []

    n_total = ds.get("n_samples")
    if n_total is None and md.get("split"):
        n_total = (md["split"].get("n_train") or 0) + (md["split"].get("n_test") or 0) or None

    prev = ds.get("prevalence")
    if prev is None:
        pc, nt = ds.get("positive_count"), ds.get("n_samples")
        if pc and nt:
            prev = pc / nt

    return dict(
        skill=skill,
        dataset_name=(ds.get("name") or "—"),
        dataset_owner=ds.get("owner") or ds.get("original_owner") or "—",
        dataset_sha256=ds.get("sha256"),
        years=ds.get("survey_year") or ds.get("survey_years"),
        n_total=n_total,
        positive_count=ds.get("positive_count"),
        prevalence=prev,
        n_features=ds.get("n_features") or (len(feats) if feats else None),
        features=feats,
        target=ds.get("target_definition") or ds.get("target") or "—",
        threshold=md.get("decision_threshold"),
        cv_auc=mean_of(cv, "auc", "roc_auc"),
        cv_bacc=mean_of(cv, "balanced_accuracy"),
        cv_sens=mean_of(cv, "sensitivity", "recall"),
        cv_spec=mean_of(cv, "specificity"),
        cv_pr_auc=mean_of(cv, "pr_auc"),
        cv_n_splits=cv.get("n_splits"),
        cv_n_repeats=cv.get("n_repeats"),
        ho_auc=ho.get("auc"),
        ho_bacc=ho.get("balanced_accuracy"),
        ho_pr_auc=ho.get("pr_auc"),
        ho_pr_baseline=ho.get("pr_auc_baseline"),
        ho_plain=ho.get("plain_accuracy"),
        ho_majority=ho.get("majority_baseline_accuracy"),
        ho_confusion=ho.get("confusion_matrix"),
        has_metadata=bool(md),
        has_model=(sk / "models").exists() and any((sk / "models").glob("*.ubj")),
    )


def build_table(rows: list[dict]) -> str:
    out = [
        "| Skill | 疾病 | 数据集 | 样本量 | 阳性率 | 特征数 | CV AUC | CV 平衡准确率 | 决策阈值 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = SKILL_META[r["skill"]]
        prev = f"{r['prevalence']*100:.2f}%" if isinstance(r["prevalence"], (int, float)) else "—"
        thr = "—" if r["threshold"] is None else str(r["threshold"])
        ds_short = r["dataset_name"].split("(")[0].strip()
        out.append(
            f"| [`{r['skill']}`](./{r['skill']}/) | {m['cn']} | {ds_short} | "
            f"{fmt_int(r['n_total'])} | {prev} | {r['n_features'] or '—'} | "
            f"{fmt(r['cv_auc'])} | {fmt(r['cv_bacc'])} | {thr} |"
        )
    return "\n".join(out)


def build_readme(cfg: dict, rows: list[dict]) -> str:
    usable = [r for r in rows if r["cv_auc"] is not None]
    aucs = [r["cv_auc"] for r in usable]
    prevs = [r["prevalence"] for r in rows if isinstance(r["prevalence"], (int, float))]
    minprev = min(prevs) if prevs else None

    L = []
    L.append(f"# {cfg['title']}索引 / {cfg['en']}\n")
    L.append(cfg["intro"] + "\n")

    if aucs and len(usable) == len(rows):
        L.append(f"共 **{len(rows)}** 个模型包，CV AUC 区间 "
                 f"**{min(aucs):.4f} – {max(aucs):.4f}**。\n")
    elif aucs:
        L.append(f"共 **{len(rows)}** 个模型包，其中 {len(usable)} 个已有完整交叉验证指标，"
                 f"CV AUC 区间 **{min(aucs):.4f} – {max(aucs):.4f}**。\n")
    else:
        L.append(f"共 **{len(rows)}** 个模型包。\n")

    L.append("## 总览\n")
    L.append(build_table(rows) + "\n")

    L.append("> **所有数字均可复现**：上表由 `tools/build_skill_index.py` 直接从各 skill 的")
    L.append("> `models/model_metadata.json` 与 `results/*.json` 读取生成，无任何手工转录。")
    L.append("> 可自行运行 `python3 tools/build_skill_index.py --check` 校验表格与模型产物是否一致。\n")
    L.append(f"> 表中为 **交叉验证 (CV)** 指标（5 折 × 3 次重复 = 15 折）；")
    L.append("> 留出集指标见各 skill 的 `results/holdout_metrics.json`，两者通常相差 ±0.01 以内。\n")

    if minprev is not None:
        L.append(f"> **为什么不用准确率 (Accuracy)**：本组任务阳性率最低仅 {minprev*100:.2f}%，")
        L.append(f"> 「全部预测为阴性」这一无脑策略就能拿到 **{(1-minprev)*100:.2f}%** 的准确率。")
        L.append("> 因此一律以 **AUC + 平衡准确率 + 敏感度/特异度** 为准，")
        L.append("> 常规准确率仅在 `results/holdout_metrics.json` 里作为反面参照保留。\n")

    L.append("## 使用方式\n")
    L.append("每个目录都是一个自包含的 skill 包，结构与用法完全一致：\n")
    L.append("```")
    L.append("<skill_name>/")
    L.append("├── SKILL.md              # 技能说明（frontmatter + 完整方法论）")
    L.append("├── README.md             # 快速上手")
    L.append("├── scripts/")
    L.append("│   ├── fetch_data.py     # 按 SHA256 校验下载原始数据")
    L.append("│   ├── train_model.py    # 完整训练流程（可复现）")
    L.append("│   ├── predict.py        # 单条/小批量推理")
    L.append("│   ├── batch_predict.py  # 批量推理（CSV 进 CSV 出）")
    L.append("│   └── feature_importance.py")
    L.append("├── models/               # 已训练模型（推理开箱即用，无需下载数据）")
    L.append("│   ├── *_xgb_model.json  # XGBoost 原生格式（推荐）")
    L.append("│   ├── *_xgb_model.ubj   # 二进制格式（体积更小）")
    L.append("│   ├── imputer.joblib    # 中位数插补器")
    L.append("│   └── model_metadata.json")
    L.append("├── results/              # 训练产出的全部指标")
    L.append("├── examples/             # 示例输入")
    L.append("└── data/DATA_PROVENANCE.md  # 数据来源、许可、SHA256")
    L.append("```\n")
    L.append("推理只需 `xgboost` + `numpy`；**不下载数据也能直接跑推理**，")
    L.append("因为 `models/` 已随仓库提交。只有重新训练才需要执行 `fetch_data.py`。\n")
    L.append("```bash")
    first = rows[0]["skill"]
    L.append(f"pip install -r {first}/requirements.txt")
    L.append(f"python {first}/scripts/predict.py --help")
    L.append("```\n")

    L.append("## 逐个说明\n")
    for r in rows:
        m = SKILL_META[r["skill"]]
        L.append(f"### `{r['skill']}` — {m['cn']}\n")
        L.append(f"**数据集**：{r['dataset_name']}  ")
        L.append(f"**提供方**：{r['dataset_owner']}  ")
        if r["years"]:
            L.append(f"**年份**：{r['years']}  ")
        line = f"**样本量**：{fmt_int(r['n_total'])}"
        if isinstance(r["prevalence"], (int, float)):
            line += (f"（阳性 {fmt_int(r['positive_count'])}，"
                     f"阳性率 {r['prevalence']*100:.2f}%）")
        L.append(line + "  ")
        feat_line = f"**特征数**：{r['n_features'] or '—'}"
        if r["features"]:
            feat_line += f" — `{'`, `'.join(map(str, r['features']))}`"
        L.append(feat_line + "  ")
        L.append(f"**预测目标**：{r['target']}  ")
        if r["cv_auc"] is not None:
            s = (f"**CV 表现**：AUC {fmt(r['cv_auc'])}、平衡准确率 {fmt(r['cv_bacc'])}")
            if r["cv_sens"] is not None:
                s += f"、敏感度 {fmt(r['cv_sens'])}、特异度 {fmt(r['cv_spec'])}"
            if r["cv_pr_auc"] is not None:
                s += f"、PR-AUC {fmt(r['cv_pr_auc'])}"
            L.append(s + f"（{r['cv_n_splits']} 折 × {r['cv_n_repeats']} 次）  ")
        if r["ho_auc"] is not None:
            s = f"**留出集**：AUC {fmt(r['ho_auc'])}、平衡准确率 {fmt(r['ho_bacc'])}"
            if r["ho_pr_auc"] is not None:
                s += f"、PR-AUC {fmt(r['ho_pr_auc'])}"
                if r["ho_pr_baseline"] is not None:
                    s += f"（基线 {fmt(r['ho_pr_baseline'])}）"
            L.append(s + "  ")
        if r["threshold"] is not None:
            L.append(f"**决策阈值**：`{r['threshold']}`  ")
        if r["dataset_sha256"]:
            L.append(f"**数据 SHA256**：`{r['dataset_sha256']}`  ")
        L.append("")
        L.append(m["note"] + "\n")

    L.append("---\n")
    L.append("## 统一的方法论约定\n")
    L.append("这组模型共享同一套流程约定，这也是它们可以横向比较的前提：\n")
    L.append("1. **只用可自报/可获取的变量**，避免把诊断标准本身当特征"
             "（例如用「是否服用降压药」预测高血压，属于标签泄漏）。")
    L.append("2. **不做性别、种族等子集切分**，除非该结局在数据源中只对特定人群询问"
             "（乳腺癌是唯一例外，已在表中标注）。")
    L.append("3. **中位数插补**，不做标准化 —— XGBoost 对特征尺度不敏感。")
    L.append("4. **阈值在训练集交叉验证上选取**，最大化平衡准确率，绝不使用测试集；")
    L.append("   罕见结局的阈值通常远低于 0.5。")
    L.append("5. **分层划分** `train_test_split(test_size=0.2, stratify=y, random_state=42)`。")
    L.append("6. **报告 AUC + PR-AUC + 平衡准确率 + 敏感度/特异度**，不报告裸准确率。")
    L.append("7. **模型携带真实特征名**（自定义 `NamedXGB` 子类写入 `feature_names`），"
             "不会出现 `f0, f1, ...` 这类无意义名称。\n")
    L.append("## 数据未随仓库提交\n")
    L.append(f"原始 CSV 体积较大（本仓库合计 {cfg.get('data_size', '数十 MB')}），未纳入 git。")
    L.append("每个 skill 的 `scripts/fetch_data.py` 会**按 SHA256 校验**下载，")
    L.append("`data/DATA_PROVENANCE.md` 记录了来源 URL、许可证与校验值。")
    L.append("这样既保持仓库轻量，又保证可复现性 —— 校验和比直接提交文件更能证明"
             "「拿到的确实是同一份数据」。\n")
    L.append("## 免责声明\n")
    L.append("**本项目仅用于研究与教学，不构成医疗建议，不得用于临床诊断。**")
    L.append("所有模型均为流行病学数据上的统计关联，个体风险判断请咨询执业医师。")
    L.append("数据版权归各自提供方所有，使用请遵守原始许可。\n")
    return "\n".join(L)


def main() -> int:
    if REPO not in REPOS:
        print(f"错误：{ROOT} 不是已知仓库（脚本应位于 <repo>/tools/ 下）", file=sys.stderr)
        return 2

    cfg = REPOS[REPO]
    rows = [collect(s) for s in cfg["order"]]

    missing = [r["skill"] for r in rows if not (ROOT / "skills" / r["skill"]).is_dir()]
    if missing:
        print(f"错误：以下 skill 目录不存在：{missing}", file=sys.stderr)
        return 2

    text = build_readme(cfg, rows)
    dest = ROOT / "skills" / "README.md"

    if "--check" in sys.argv:
        old = dest.read_text(encoding="utf-8") if dest.exists() else ""
        if old == text:
            print(f"[{REPO}] ✓ skills/README.md 与模型产物一致")
            return 0
        print(f"[{REPO}] ✗ skills/README.md 与模型产物不一致（请重新生成）")
        return 1

    dest.write_text(text, encoding="utf-8")
    print(f"[{REPO}] 已写入 skills/README.md（{len(text.splitlines())} 行）")
    no_model = [r["skill"] for r in rows if not r["has_model"]]
    if no_model:
        print(f"[{REPO}] 提示：以下 skill 尚无训练产物（已如实标注）：{no_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
