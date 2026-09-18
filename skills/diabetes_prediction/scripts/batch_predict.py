#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测 - 批量推理脚本
================================================================
对 CSV 中的多个患者批量预测糖尿病风险，输出带概率与风险等级的结果。

用法:
    python scripts/batch_predict.py --input patients.csv --output results.csv
    python scripts/batch_predict.py --input patients.csv --output results.json --format json

输入 CSV 必需列（顺序无关，兼容 Pima 原始列名 Glucose/BMI/Outcome 等）:
    preg,plas,pres,skin,insu,mass,pedi,age

说明: plas/pres/skin/insu/mass 中的 0 会被视为缺失并用中位数插补，
      与训练时的预处理保持一致。
================================================================
"""
import argparse
import json
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict import (  # noqa: E402
    FEATURE_ORDER,
    ZERO_AS_MISSING,
    load_artifacts,
    predict_patient,
)

COLUMN_ALIASES = {
    "pregnancies": "preg", "glucose": "plas", "bloodpressure": "pres",
    "blood_pressure": "pres", "skinthickness": "skin", "skin_thickness": "skin",
    "insulin": "insu", "bmi": "mass",
    "diabetespedigreefunction": "pedi", "diabetes_pedigree_function": "pedi",
    "outcome": "label", "class": "label", "target": "label",
}


def batch_predict(input_path, output_path=None, out_format="csv",
                  model=None, medians=None, metadata=None):
    """批量预测，返回结果 DataFrame。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"错误: 输入文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(
            f"错误: 输入 CSV 缺少必需列: {missing}\n必需列: {FEATURE_ORDER}"
        )

    if model is None or medians is None:
        model, medians, metadata, _ = load_artifacts()

    rows = []
    for _, row in df.iterrows():
        base = {f: row[f] for f in FEATURE_ORDER}
        try:
            r = predict_patient(**base, model=model, medians=medians, metadata=metadata)
            rows.append({
                **base,
                "probability": r["probability"],
                "prediction": r["prediction"],
                "label": r["label"],
                "risk_level": r["risk_level"],
                "advice": r["advice"],
                "n_imputed": len(r["imputed_features"]),
                "status": "ok",
            })
        except (ValueError, TypeError) as e:
            rows.append({
                **base,
                "probability": None, "prediction": None, "label": None,
                "risk_level": None, "advice": None, "n_imputed": None,
                "status": f"error: {e}",
            })

    result = pd.DataFrame(rows)

    if output_path:
        outdir = os.path.dirname(os.path.abspath(output_path))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        if out_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(orient="records"), f,
                          ensure_ascii=False, indent=2)
        else:
            result.to_csv(output_path, index=False, encoding="utf-8-sig")

    return result


def main():
    parser = argparse.ArgumentParser(description="糖尿病风险预测（批量）")
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="输出格式")
    args = parser.parse_args()

    try:
        result = batch_predict(args.input, args.output, args.format)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    total = len(result)
    ok_mask = result["status"] == "ok"
    ok = int(ok_mask.sum())

    print("=" * 64)
    print(f"批量预测完成: 共 {total} 条，成功 {ok} 条，失败 {total - ok} 条")
    print("=" * 64)
    if ok:
        show = result.loc[ok_mask, FEATURE_ORDER + ["probability", "label", "risk_level"]]
        print(show.to_string(index=False))
        print("\n风险等级分布:")
        for level, cnt in result.loc[ok_mask, "risk_level"].value_counts().items():
            print(f"  {level}: {cnt}")
        n_imp = int((result.loc[ok_mask, "n_imputed"] > 0).sum())
        if n_imp:
            print(f"提示: {n_imp} 条记录含 0 值，已按中位数插补"
                  f"（涉及列: {', '.join(ZERO_AS_MISSING)}）")
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
