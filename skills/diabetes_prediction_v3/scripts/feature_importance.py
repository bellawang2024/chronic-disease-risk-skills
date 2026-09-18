#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测 - 特征重要性查看脚本（v3）
================================================================
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
    "age": "年龄", "bmi": "BMI", "height_cm": "身高", "weight_kg": "体重",
    "smoker": "吸烟状态", "hypertension": "高血压", "high_chol": "高胆固醇",
}


def load_importance():
    path = os.path.join(RESULTS_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "results/feature_importance.json"

    import xgboost as xgb

    model_path = os.path.join(MODELS_DIR, "diabetes_xgb_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("错误: 未找到模型或特征重要性结果，请先运行 train_model.py")
    features = list(FEATURE_LIMITS)
    booster = xgb.Booster()
    booster.load_model(model_path)
    gain = booster.get_score(importance_type="gain")
    keys = list(gain.keys())
    if keys and all(re.fullmatch(r"f\d+", k) for k in keys):
        gain = {features[int(k[1:])]: v for k, v in gain.items() if int(k[1:]) < len(features)}
    total = sum(gain.values()) or 1.0
    data = sorted([{"feature": f, "importance": round(gain.get(f, 0.0) / total, 4)}
                   for f in features], key=lambda d: d["importance"], reverse=True)
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

    print("=" * 76)
    print("     影响糖尿病的特征（按模型重要性降序）")
    print("=" * 76)
    print(f"来源: {source}\n")
    print(f"{'排序':<5}{'特征':<14}{'中文名':<10}{'说明':<26}{'重要性':>8}")
    print("-" * 76)
    for i, item in enumerate(data, 1):
        f = item["feature"]
        desc = FEATURE_LIMITS.get(f, (0, 0, ""))[2]
        print(f"{i:<5}{f:<14}{FEATURE_NAMES_CN.get(f, f):<10}{desc:<26}{item['importance']:>8.4f}")
    print("=" * 76)
    print("⚠️  重要解读警告 —— 请勿把上表当作【风险因素强度】排序：")
    print("   1) 树模型的 gain 重要性偏向二值特征（高血压/高胆固醇只需一次分裂即可分开人群），")
    print("      而年龄是连续变量，其信号被与之相关的高血压/高胆固醇大量吸收。")
    print("      实测：去掉高血压/高胆固醇后，年龄重要性从 0.068 升到 0.630、BMI 从 0.034 升到 0.221。")
    print("   2) 单变量实测的患病率梯度（更能反映真实风险强度）：")
    print("        年龄 18-29 岁 1.14% → 70-85 岁 19.46%（17 倍）")
    print("        BMI 10-25 → 5.14%，BMI≥35 → 20.60%（4 倍）")
    print("        无高血压 4.43% → 有高血压 22.07%；无高胆固醇 5.49% → 有 22.52%")
    print("   3) 本数据为横断面调查，高血压/高胆固醇与糖尿病的强关联中，")
    print("      既含真实的代谢综合征关联，也含【确诊糖尿病者被更多筛查从而查出这些病】的")
    print("      检出偏倚成分，二者无法在本设计下分离。")

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
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(labels, values, color="#1F6FB2")
        ax.set_ylabel("importance (gain)")
        ax.set_title("Diabetes Risk - Feature Importance (XGBoost, NHIS 2016-2025)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\n图表已保存: {os.path.abspath(args.plot)}")


if __name__ == "__main__":
    main()
