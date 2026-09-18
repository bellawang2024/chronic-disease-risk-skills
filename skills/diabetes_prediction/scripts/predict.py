#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测 - 单患者推理脚本
================================================================
加载 models/ 下的推理工件，对单个患者做糖尿病风险二分类预测。

预处理与训练完全一致：
  1. 生理上不可能为 0 的列（plas/pres/skin/insu/mass）中，0 视为缺失
  2. 缺失值用训练集（发布模型为全量）中位数插补
  3. XGBoost 预测正类概率，阈值 0.5 判定

用法:
    python scripts/predict.py --preg 2 --plas 120 --pres 70 --skin 25 \
        --insu 150 --mass 30 --pedi 0.5 --age 35
    python scripts/predict.py --preg 2 --plas 120 --pres 70 --skin 0 \
        --insu 0 --mass 30 --pedi 0.5 --age 35 --json

作为模块调用:
    from predict import predict_patient
    predict_patient(preg=2, plas=120, pres=70, skin=25,
                    insu=150, mass=30, pedi=0.5, age=35)
================================================================
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(SKILL_DIR, "models")

FEATURE_ORDER = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]

# 生理上不可能为 0 的列（0 视为缺失）
ZERO_AS_MISSING = ["plas", "pres", "skin", "insu", "mass"]

# 合理取值范围（依据数据集实际分布留出余量）
FEATURE_LIMITS = {
    "preg": (0, 20, "怀孕次数(次)"),
    "plas": (0, 300, "口服糖耐量试验2小时血糖(mg/dL)"),
    "pres": (0, 200, "舒张压(mm Hg)"),
    "skin": (0, 120, "三头肌皮褶厚度(mm)"),
    "insu": (0, 1000, "2小时血清胰岛素(mu U/ml)"),
    "mass": (0.0, 80.0, "BMI体重指数(kg/m^2)"),
    "pedi": (0.0, 3.0, "糖尿病家族遗传函数"),
    "age": (1, 120, "年龄(岁)"),
}

RISK_BANDS = [
    (0.30, "低风险", "继续保持健康的生活方式，定期体检"),
    (0.70, "中等风险", "建议控制饮食与体重、增加运动，并定期复查血糖"),
    (1.01, "高风险", "强烈建议尽快就医，进行空腹血糖 / 糖耐量等专项检查"),
]


# ---------------------------------------------------------------------------
# 工件加载
# ---------------------------------------------------------------------------
def _load_metadata():
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_medians(metadata):
    """优先使用元数据中记录的中位数（无需 joblib），否则退回 joblib 工件。"""
    stats = ((metadata or {}).get("preprocessing", {}) or {}).get("imputer_statistics")
    if stats:
        medians = np.array([float(stats[c]) for c in FEATURE_ORDER], dtype=float)
        return medians, "metadata:imputer_statistics"

    try:
        import joblib

        path = os.path.join(MODELS_DIR, "imputer.joblib")
        if os.path.exists(path):
            imputer = joblib.load(path)
            return np.asarray(imputer.statistics_, dtype=float), "imputer.joblib"
    except ImportError:
        pass

    raise FileNotFoundError(
        f"错误: 未找到插补参数。请先运行 train_model.py。查找路径: {MODELS_DIR}"
    )


def _load_model():
    """优先 XGBoost 原生 json/ubj 工件。"""
    import xgboost as xgb

    for name in ("diabetes_xgb_model.json", "diabetes_xgb_model.ubj"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            booster = xgb.Booster()
            booster.load_model(path)
            return booster, name

    raise FileNotFoundError(
        f"错误: 未找到模型工件。请先运行:\n  python {os.path.join(SCRIPT_DIR, 'train_model.py')}"
    )


def load_artifacts():
    """返回 (model, medians, metadata, sources)。"""
    metadata = _load_metadata()
    model, model_src = _load_model()
    medians, median_src = _load_medians(metadata)
    return model, medians, metadata, {"model": model_src, "imputer": median_src}


# ---------------------------------------------------------------------------
# 预处理与推理
# ---------------------------------------------------------------------------
def impute_row(values, medians):
    """把 0（缺失）替换为中位数，返回与训练一致的二维数组。"""
    row = np.array([float(values[n]) for n in FEATURE_ORDER], dtype=float)
    for i, name in enumerate(FEATURE_ORDER):
        if name in ZERO_AS_MISSING and row[i] == 0:
            row[i] = medians[i]
    return row.reshape(1, -1)


def _positive_proba(model, X):
    import xgboost as xgb

    if isinstance(model, xgb.Booster):
        dm = xgb.DMatrix(np.asarray(X, dtype=float), feature_names=FEATURE_ORDER)
        return float(np.asarray(model.predict(dm)).reshape(-1)[0])

    return float(model.predict_proba(X)[0][1])


def risk_band(probability):
    for threshold, level, advice in RISK_BANDS:
        if probability < threshold:
            return level, advice
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict_patient(preg, plas, pres, skin, insu, mass, pedi, age,
                    model=None, medians=None, metadata=None):
    """对单个患者进行糖尿病风险预测，返回结构化 dict。"""
    values = {"preg": preg, "plas": plas, "pres": pres, "skin": skin,
              "insu": insu, "mass": mass, "pedi": pedi, "age": age}

    for name in FEATURE_ORDER:
        if values[name] is None:
            raise ValueError(f"错误: 缺少必要特征 '{name}'")
        lo, hi, desc = FEATURE_LIMITS[name]
        v = float(values[name])
        if not np.isfinite(v):
            raise ValueError(f"错误: 特征 '{name}' 取值非法: {values[name]}")
        if v < lo or v > hi:
            raise ValueError(
                f"错误: 特征 '{name}' ({desc}) = {values[name]} 超出合理范围 [{lo}, {hi}]"
            )

    if model is None or medians is None:
        model, medians, metadata, sources = load_artifacts()
    else:
        sources = {"model": "injected", "imputer": "injected"}

    metadata = metadata or {}
    threshold = float(metadata.get("decision_threshold", 0.5))

    X = impute_row(values, medians)
    probability = _positive_proba(model, X)
    prediction = int(probability >= threshold)
    level, advice = risk_band(probability)

    imputed = [n for n in FEATURE_ORDER
               if n in ZERO_AS_MISSING and float(values[n]) == 0]

    return {
        "probability": round(probability, 4),
        "probability_pct": f"{probability:.2%}",
        "prediction": prediction,
        "label": "糖尿病" if prediction == 1 else "非糖尿病",
        "risk_level": level,
        "advice": advice,
        "decision_threshold": threshold,
        "features": {n: float(values[n]) for n in FEATURE_ORDER},
        "imputed_features": imputed,
        "model": metadata.get("model_type", "xgboost.XGBClassifier"),
        "model_version": metadata.get("version"),
        "model_trained_at": metadata.get("trained_at"),
        "_sources": sources,
    }


def _print_report(result, sources):
    print("\n" + "=" * 60)
    print("                糖尿病风险预测结果")
    print("=" * 60)
    print(f"\n风险概率: {result['probability_pct']}   (判定阈值 {result['decision_threshold']})")
    print(f"预测结果: {'阳性 - 糖尿病' if result['prediction'] == 1 else '阴性 - 非糖尿病'}")
    print(f"风险等级: {result['risk_level']}")
    print(f"建议: {result['advice']}")
    print("\n输入特征:")
    for k, v in result["features"].items():
        mark = "  [0 -> 中位数插补]" if k in result["imputed_features"] else ""
        print(f"  {k:5s} = {v}{mark}")
    print(f"\n模型工件: {sources['model']}  |  插补参数: {sources['imputer']}")
    print("=" * 60)
    print("提示: 本结果仅供健康风险筛查参考，不能替代专业医生诊断。")


def main():
    parser = argparse.ArgumentParser(description="糖尿病风险预测（单患者）")
    parser.add_argument("--preg", type=float, required=True, help="怀孕次数")
    parser.add_argument("--plas", type=float, required=True, help="血糖浓度 (mg/dL)")
    parser.add_argument("--pres", type=float, required=True, help="舒张压 (mm Hg)")
    parser.add_argument("--skin", type=float, required=True, help="三头肌皮褶厚度 (mm)")
    parser.add_argument("--insu", type=float, required=True, help="血清胰岛素 (mu U/ml)")
    parser.add_argument("--mass", type=float, required=True, help="BMI 体重指数")
    parser.add_argument("--pedi", type=float, required=True, help="糖尿病家族遗传函数")
    parser.add_argument("--age", type=float, required=True, help="年龄")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    try:
        result = predict_patient(
            preg=args.preg, plas=args.plas, pres=args.pres, skin=args.skin,
            insu=args.insu, mass=args.mass, pedi=args.pedi, age=args.age,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result, result["_sources"])


if __name__ == "__main__":
    main()
