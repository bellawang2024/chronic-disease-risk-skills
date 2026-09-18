#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测 - 批量推理脚本（v3，全人群）
================================================================
用法:
    python scripts/batch_predict.py --input people.csv --output results.csv
    python scripts/batch_predict.py --input people.csv --output r.json --format json

输入 CSV 必需列: age, bmi, smoker
可选列: height_cm, weight_kg, hypertension, high_chol
================================================================
"""
import argparse
import json
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict import (  # noqa: E402
    CORE_FEATURES,
    FEATURE_ORDER,
    load_artifacts,
    predict_patient,
)

COLUMN_ALIASES = {
    "height": "height_cm", "weight": "weight_kg", "bmi_kg_m2": "bmi",
    "smoking": "smoker", "hbp": "hypertension", "bp": "hypertension",
    "highchol": "high_chol", "highcholesterol": "high_chol", "cholesterol": "high_chol",
}


def batch_predict(input_path, output_path=None, out_format="csv", threshold=None,
                  model=None, medians=None, order=None, metadata=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"错误: 输入文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})

    if model is None or medians is None:
        model, medians, order, metadata, _ = load_artifacts()
    order = order or FEATURE_ORDER

    core = [c for c in CORE_FEATURES if c in order]
    missing = [c for c in core if c not in df.columns]
    if missing:
        raise ValueError(
            f"错误: 输入 CSV 缺少核心列: {missing}\n"
            f"核心列(必需): {core}\n"
            f"可选列(可省略，将用中位数插补): {[f for f in order if f not in core]}"
        )

    rows = []
    for _, row in df.iterrows():
        base = {}
        for f in order:
            if f not in df.columns:
                base[f] = None
                continue
            v = row[f]
            base[f] = None if pd.isna(v) else float(v)
        summary = {c: base.get(c) for c in core}
        try:
            r = predict_patient(**base, model=model, medians=medians, order=order,
                                metadata=metadata, threshold=threshold)
            rows.append({
                **summary,
                "probability": r["probability"],
                "risk_level": r["risk_level"],
                "above_threshold": r["prediction"],
                "n_imputed": len(r["imputed_features"]),
                "status": "ok",
            })
        except (ValueError, TypeError) as e:
            rows.append({
                **summary,
                "probability": None, "risk_level": None, "above_threshold": None,
                "n_imputed": None, "status": f"error: {e}",
            })

    result = pd.DataFrame(rows)

    if output_path:
        outdir = os.path.dirname(os.path.abspath(output_path))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        if out_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        else:
            result.to_csv(output_path, index=False, encoding="utf-8-sig")

    return result


def main():
    parser = argparse.ArgumentParser(description="糖尿病风险预测（全人群，批量）")
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--threshold", type=float, default=None, help="自定义判定阈值")
    args = parser.parse_args()

    try:
        result = batch_predict(args.input, args.output, args.format, args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    total = len(result)
    ok_mask = result["status"] == "ok"
    ok = int(ok_mask.sum())

    print("=" * 72)
    print(f"批量预测完成: 共 {total} 条，成功 {ok} 条，失败 {total - ok} 条")
    print("=" * 72)
    if ok:
        print(result.loc[ok_mask, CORE_FEATURES + ["probability", "risk_level"]]
              .to_string(index=False))
        print("\n风险等级分布:")
        for level, cnt in result.loc[ok_mask, "risk_level"].value_counts().items():
            print(f"  {level}: {cnt}")
        n_imp = int((result.loc[ok_mask, "n_imputed"] > 0).sum())
        if n_imp:
            print(f"提示: {n_imp} 条记录含未提供的可选指标，已用训练中位数插补")
        print("\n⚠️  模型仅用风险因素，不含生化指标；确诊须依靠血糖 / HbA1c 检测。")
    if total - ok:
        print("\n失败明细:")
        for _, r in result.loc[~ok_mask].iterrows():
            print(f"  {r['status']}")
    if args.output:
        print(f"\n结果已写入: {os.path.abspath(args.output)}")
    else:
        print("\n提示: 未指定 --output，结果仅打印到终端。")


if __name__ == "__main__":
    main()
