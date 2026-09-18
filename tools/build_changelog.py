#!/usr/bin/env python3
"""
从各 skill 的模型产物生成 CHANGELOG.md。

与 tools/build_skill_index.py 同一原则：文件里的每个数字都来自
model_metadata.json / results/*.json，不做手工转录。

本脚本位于 <repo>/tools/，运行时会自动识别所在仓库。

运行：  python3 tools/build_changelog.py
校验：  python3 tools/build_changelog.py --check
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.name

CFG = {
    "chronic-disease-risk-skills": dict(
        cn="慢病风险预测 Skills 技能包",
        repo_cn="慢病",
        data_size="约 69 MB",
        order=[
            ("chd_prediction", "冠心病 / 心肌梗死", "BRFSS 2017"),
            ("copd_prediction", "慢性阻塞性肺疾病 (COPD)", "BRFSS 2017"),
            ("stroke_prediction", "脑卒中", "BRFSS 2017"),
            ("hypertension_prediction", "高血压", "NHANES 2017–2018"),
            ("diabetes_prediction_v3", "糖尿病 (v3)", "NHIS 2016–2025"),
            ("hyperlipidemia_prediction", "高脂血症", "NHIS 2016–2025"),
            ("diabetes_prediction", "糖尿病 (v2 · Pima)", "Pima Indians"),
        ],
        docs=["docs/METHODOLOGY.md（方法论）", "docs/DATA_SOURCES.md（数据来源与选型调研）",
              "docs/PERFORMANCE.md（全部技能包性能指标）"],
    ),
    "cancer-risk-skills": dict(
        cn="癌症风险预测 Skills 技能包",
        repo_cn="癌症",
        data_size="约 18.5 MB",
        order=[
            ("lung_cancer_prediction", "肺癌", "NHIS 2016–2025"),
            ("colorectal_cancer_prediction", "结直肠癌", "NHIS 2016–2025"),
            ("breast_cancer_prediction", "乳腺癌（仅女性）", "NHIS 2016–2025"),
            ("thyroid_cancer_prediction", "甲状腺癌", "NHIS 2016–2025"),
            ("gastric_cancer_prediction", "胃癌（草稿）", "—"),
        ],
        docs=["docs/RARE_OUTCOMES.md（稀有结局评估方法论）",
              "docs/DATA_AVAILABILITY.md（11 个数据源可及性调研）",
              "docs/PERFORMANCE.md（全部技能包性能指标）"],
    ),
}


def load(p: pathlib.Path):
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def mean_of(d, *keys):
    for k in keys:
        if k in d:
            v = d[k]
            return v["mean"] if isinstance(v, dict) else v
    return None


def collect(skill):
    base = ROOT / "skills" / skill
    md = load(base / "models" / "model_metadata.json") or {}
    ds = md.get("dataset", {}) or {}
    cv = load(base / "results" / "cv_metrics.json")
    cv = cv if isinstance(cv, dict) else {}
    prev = ds.get("prevalence")
    if prev is None:
        pc, nt = ds.get("positive_count"), ds.get("n_samples")
        if pc and nt:
            prev = pc / nt
    return dict(
        n=ds.get("n_samples"), prev=prev,
        auc=mean_of(cv, "auc", "roc_auc"),
        bacc=mean_of(cv, "balanced_accuracy"),
        nf=ds.get("n_features"),
        thr=md.get("decision_threshold"),
        draft=not (base / "models" / "model_metadata.json").exists(),
    )


def build(cfg) -> str:
    L = []
    L.append("# 更新日志 / Changelog\n")
    L.append("本项目所有重要变更都记录在此文件。")
    L.append("格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，"
             "版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。\n")
    L.append("---\n")
    L.append("## [1.0.0] — 2026-09-18\n")

    L.append(f"### 新增 — {cfg['cn']}\n")
    L.append(f"{cfg['repo_cn']}方向的首个正式版本，交付 **{len(cfg['order'])} 个 Agent Skills 技能包**，"
             "全部构建在公开可获取的人群调查数据之上。\n")
    L.append("| 技能包 | 疾病 | 数据来源 | 样本量 | 阳性率 | 特征数 | CV AUC | CV 平衡准确率 | 决策阈值 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for skill, disease, source in cfg["order"]:
        c = collect(skill)
        if c["draft"]:
            L.append(f"| `{skill}` | {disease} | {source} | — | — | — | — | — | — |")
            continue
        p = f"{c['prev']*100:.2f}%" if isinstance(c["prev"], (int, float)) else "—"
        n = f"{c['n']:,}" if isinstance(c["n"], int) else "—"
        au = f"{c['auc']:.4f}" if c["auc"] is not None else "—"
        ba = f"{c['bacc']:.4f}" if c["bacc"] is not None else "—"
        L.append(f"| `{skill}` | {disease} | {source} | {n} | {p} | {c['nf'] or '—'} | "
                 f"{au} | {ba} | {c['thr']} |")
    L.append("")

    L.append("每个技能包均包含：\n")
    for item in ("`SKILL.md` — 技能说明（frontmatter + 完整方法论）",
                 "`scripts/fetch_data.py` — 按 SHA256 校验下载原始数据",
                 "`scripts/train_model.py` — 完整可复现训练流程",
                 "`scripts/predict.py` / `batch_predict.py` — 推理脚本",
                 "`models/` — 已训练模型（推理开箱即用）",
                 "`results/` — 交叉验证与留出集完整指标",
                 "`data/DATA_PROVENANCE.md` — 数据来源、许可证、校验值"):
        L.append(f"- {item}")
    L.append("")

    L.append("### 新增 — 工具与文档\n")
    L.append("- `tools/build_skill_index.py` — 从模型产物自动生成 `skills/README.md` 索引，"
             "支持 `--check` 校验表格与模型是否一致")
    L.append("- `tools/build_changelog.py` — 生成本文件")
    for d in cfg["docs"]:
        L.append(f"- {d}")
    L.append("- `CITATION.cff`、`LICENSE`（MIT）、`NOTICE.md`（数据说明与医疗免责）、`CONTRIBUTING.md`")
    L.append("")

    L.append("### 变更\n")
    L.append(f"- 交付物定位由「模型集」调整为「**Agent Skills 技能包**」，"
             f"仓库更名为 `{REPO}`")
    L.append("- 中英文 README、`CITATION.cff`、各文档标题同步更新，"
             "明确最终交付物为技能包、且全部基于公开数据")
    L.append("")

    L.append("### 说明\n")
    L.append(f"- 原始数据集 CSV **不纳入 git**（本仓库合计 {cfg['data_size']}）。"
             "改由各技能包的 `scripts/fetch_data.py` 按 SHA256 校验下载 —— "
             "校验和比直接提交文件更能证明「拿到的确实是同一份数据」")
    L.append("- `docs/PERFORMANCE.md` 与 `skills/README.md` 中的全部指标均由模型产物自动生成，"
             "无手工转录")
    L.append("- 模型中的特征名均为真实变量名（自定义 `NamedXGB` 子类写入 `feature_names`），"
             "不会出现 `f0, f1, ...`")
    L.append("")
    L.append("---\n")
    L.append("## 免责声明\n")
    L.append("**本项目仅用于研究与教学，不构成医疗建议，不得用于临床诊断。**")
    L.append("所有模型反映的都是流行病学数据上的统计关联，个体风险判断请咨询执业医师。\n")
    return "\n".join(L)


def main() -> int:
    if REPO not in CFG:
        print(f"错误：{ROOT} 不是已知仓库（脚本应位于 <repo>/tools/ 下）", file=sys.stderr)
        return 2
    text = build(CFG[REPO])
    dest = ROOT / "CHANGELOG.md"
    if "--check" in sys.argv:
        old = dest.read_text(encoding="utf-8") if dest.exists() else ""
        if old == text:
            print(f"[{REPO}] ✓ CHANGELOG.md 与模型产物一致")
            return 0
        print(f"[{REPO}] ✗ CHANGELOG.md 与模型产物不一致")
        return 1
    dest.write_text(text, encoding="utf-8")
    print(f"[{REPO}] 已写入 CHANGELOG.md（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
