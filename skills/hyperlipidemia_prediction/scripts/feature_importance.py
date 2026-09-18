#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高脂血症风险预测 - 特征重要性查看脚本
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
    "smoker": "吸烟状态", "hypertension": "高血压", "diabetes": "糖尿病",
}


def load_importance():
    path = os.path.join(RESULTS_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "results/feature_importance.json"

    import xgboost as xgb
    model_path = os.path.join(MODELS_DIR, "hyperlipidemia_xgb_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "错误: 未找到模型或特征重要性结果，请先运行 train_model.py")
    features = list(FEATURE_LIMITS)
    booster = xgb.Booster()
    booster.load_model(model_path)
    gain = booster.get_score(importance_type="gain")
    keys = list(gain.keys())
    # 兼容早期用 numpy 训练、特征名退化为 f0/f1... 的工件
    if keys and all(re.fullmatch(r"f\d+", k) for k in keys):
        gain = {features[int(k[1:])]: v for k, v in gain.items()
                if int(k[1:]) < len(features)}
    total = sum(gain.values()) or 1.0
    data = sorted([{"feature": f, "importance": round(gain.get(f, 0.0) / total, 4)}
                   for f in features], key=lambda d: d["importance"], reverse=True)
    return data, "现场重算 (gain 归一化)"


def main():
    parser = argparse.ArgumentParser(description="查看高脂血症模型特征重要性")
    parser.add_argument("--plot", default=None, help="可选：输出柱状图 PNG 路径")
    args = parser.parse_args()

    try:
        data, source = load_importance()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("     影响高脂血症的特征（按模型重要性降序）")
    print("=" * 78)
    print(f"来源: {source}\n")
    print(f"{'排序':<5}{'特征':<12}{'中文名':<10}{'说明':<30}{'重要性':>8}")
    print("-" * 78)
    for i, item in enumerate(data, 1):
        f = item["feature"]
        desc = FEATURE_LIMITS.get(f, (0, 0, ""))[2]
        print(f"{i:<5}{f:<12}{FEATURE_NAMES_CN.get(f, f):<10}{desc:<30}"
              f"{item['importance']:>8.4f}")
    print("=" * 78)
    print("⚠️  重要解读警告 —— 请勿把上表当作【风险因素强度】排序：")
    print("   1) 树模型的 gain 重要性偏向二值特征（高血压/糖尿病只需一次分裂")
    print("      即可分开人群），而年龄、BMI 是连续变量，其信号被大量吸收。")
    print("      实测：去掉高血压与糖尿病后，")
    print("        年龄重要性 0.247 → 0.797，BMI 0.013 → 0.066，吸烟 0.017 → 0.103。")
    print("   2) 单变量实测的患病率梯度（更能反映真实风险强度，人群基线 31.56%）：")
    print("        年龄  18-30 岁 5.09% → 70-85 岁 53.86%（11 倍）")
    print("        BMI   <20 17.09% → ≥35 38.64%（2.3 倍）")
    print("        高血压 无 18.54% → 有 54.19%（2.92 倍）")
    print("        糖尿病 无 27.15% → 有 65.42%（2.41 倍）")
    print("   3) 【反直觉】已戒烟者的高胆固醇率（42.17%）高于从不吸烟者（27.52%）。")
    print("      这不是「戒烟有害」，而是横断面设计下的反向因果与检出偏倚：")
    print("      因健康原因戒烟者、以及因确诊而改变生活方式者，更可能已被检测。")
    print("   4) 高血压/糖尿病与高脂血症同属【代谢综合征组分】，共享上游病因")
    print("      （肥胖、胰岛素抵抗），并非单向因果。但高脂血症**无症状**，")
    print("      「曾被医生告知」的前提是先做过血脂检测；而常因高血压/糖尿病")
    print("      就诊者，被查血脂的机会显著更高——因此这 2.9 / 2.4 倍中")
    print("      同时含真实的代谢关联与【检出偏倚】，本设计无法分离。")
    print("   5) 本模型性能上限（AUC 0.7956）主要由两点决定：")
    print("      (a) 标签检出依赖：无症状疾病，诊断取决于是否查过血脂；")
    print("      (b) 最强决定因素不可得：血脂值本身（循环）、高胆固醇家族史")
    print("          （NHIS 从未询问）、膳食、体力活动、社会经济地位、遗传。")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        outdir = os.path.dirname(os.path.abspath(args.plot))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        labels = [f"{d['feature']}\n{FEATURE_NAMES_CN.get(d['feature'], '')}"
                  for d in data]
        values = [d["importance"] for d in data]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(labels, values, color="#A6511F")
        ax.set_ylabel("importance (gain)")
        ax.set_title("Hyperlipidemia Risk - Feature Importance "
                     "(XGBoost, NHIS 2016-2025)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\n图表已保存: {os.path.abspath(args.plot)}")


if __name__ == "__main__":
    main()
