#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测 - 特征重要性查看脚本
================================================================
读取 results/feature_importance.json，或从模型现场重算。

用法:
    python scripts/feature_importance.py
    python scripts/feature_importance.py --plot results/feature_importance.png
================================================================
"""
import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(SKILL_DIR, "models")
RESULTS_DIR = os.path.join(SKILL_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from predict import FEATURE_LIMITS  # noqa: E402

FEATURE_NAMES_CN = {
    "preg": "怀孕次数", "plas": "血糖浓度", "pres": "舒张压", "skin": "皮褶厚度",
    "insu": "血清胰岛素", "mass": "BMI", "pedi": "家族遗传函数", "age": "年龄",
}


def load_importance():
    path = os.path.join(RESULTS_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "results/feature_importance.json"

    import xgboost as xgb

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    model_path = os.path.join(MODELS_DIR, "diabetes_xgb_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("错误: 未找到模型或特征重要性结果，请先运行 train_model.py")

    features = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            features = json.load(f).get("dataset", {}).get("features", features)

    booster = xgb.Booster()
    booster.load_model(model_path)
    gain = booster.get_score(importance_type="gain")
    # 兼容模型内部特征名为 f0..fN 的情况（按位置映射回真实列名）
    keys = list(gain.keys())
    if keys and all(re.fullmatch(r"f\d+", k) for k in keys):
        gain = {features[int(k[1:])]: v for k, v in gain.items() if int(k[1:]) < len(features)}
    total = sum(gain.values()) or 1.0
    data = sorted(
        [{"feature": f, "importance": round(gain.get(f, 0.0) / total, 4)} for f in features],
        key=lambda d: d["importance"], reverse=True,
    )
    return data, "现场重算 (gain 归一化)"


def main():
    parser = argparse.ArgumentParser(description="查看糖尿病模型特征重要性")
    parser.add_argument("--plot", default=None, help="可选：输出柱状图 PNG 路径")
    args = parser.parse_args()

    try:
        data, source = load_importance()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("=" * 74)
    print("     影响糖尿病的关键特征（按重要性降序）")
    print("=" * 74)
    print(f"来源: {source}\n")
    print(f"{'排序':<5}{'特征':<8}{'中文名':<14}{'说明':<32}{'重要性':>8}")
    print("-" * 74)
    for i, item in enumerate(data, 1):
        f = item["feature"]
        desc = FEATURE_LIMITS.get(f, (0, 0, ""))[2]
        print(f"{i:<5}{f:<8}{FEATURE_NAMES_CN.get(f, f):<14}{desc:<32}{item['importance']:>8.4f}")
    print("=" * 74)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        outdir = os.path.dirname(os.path.abspath(args.plot))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        labels = [f"{d['feature']}\n{FEATURE_NAMES_CN.get(d['feature'], '')}" for d in data]
        values = [d["importance"] for d in data]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, values, color="#2E86C1")
        ax.set_ylabel("importance")
        ax.set_title("Diabetes Risk - Feature Importance (XGBoost)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"图表已保存: {os.path.abspath(args.plot)}")


if __name__ == "__main__":
    main()
