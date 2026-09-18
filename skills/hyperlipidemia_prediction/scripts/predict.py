#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高脂血症（高胆固醇）风险预测 - 单个体推理脚本
================================================================
加载 models/ 下的推理工件，对个体做高脂血症风险二分类预测。

预处理与训练一致：缺失特征用训练中位数插补，XGBoost 输出概率，
再用**训练集交叉验证选定的阈值**判定（默认 0.305，不是 0.5）。

⚠️ 说明:
   1. 本模型**不含任何血脂检验值**（无总胆固醇、无 LDL-C、无甘油三酯）。
      原因是这些指标本身即高脂血症的诊断依据，用其预测近乎循环论证。
      本模型用于**风险分层**，确诊仍需依靠血脂检测，请遵医嘱。
   2. 标签是「曾被医生告知高胆固醇」——高脂血症**无症状**，
      「被告知」的前提是先做过血脂检测。因此本模型部分反映的是
      **医疗可及性/检测机会**，而非纯粹的生物学风险。
      这是本模型性能上限的主要来源（详见 SKILL.md）。
   3. 标签口径仅覆盖**胆固醇**，不覆盖「单纯高甘油三酯血症」。

用法:
    python scripts/predict.py --age 55 --bmi 28.5 --smoker 4
    python scripts/predict.py --age 55 --bmi 28.5 --smoker 4 \
        --height 170 --weight 82 --hypertension 1 --diabetes 0

作为模块调用:
    from predict import predict_patient
    predict_patient(age=55, bmi=28.5, smoker=4, hypertension=1, diabetes=0)
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

FEATURE_ORDER = ["age", "bmi", "height_cm", "weight_kg", "smoker",
                 "hypertension", "diabetes"]

# 核心指标：年龄/BMI/吸烟必须显式提供（不能靠中位数默认成"无风险"）
CORE_FEATURES = ["age", "bmi", "smoker"]
# 强烈建议提供：这两项是模型的两个最强特征（重要性 0.53 / 0.18），
# 缺失时会用中位数（=0，"无该病"）插补，系统性低估风险
STRONGLY_RECOMMENDED = ["hypertension", "diabetes"]
OTHER_OPTIONAL = ["height_cm", "weight_kg"]

FEATURE_LIMITS = {
    "age": (18, 85, "年龄(岁)"),
    "bmi": (10.0, 90.0, "BMI体重指数(kg/m^2)"),
    "height_cm": (120.0, 230.0, "身高(cm)"),
    "weight_kg": (30.0, 300.0, "体重(kg)"),
    "smoker": (1, 4, "吸烟状态(1=每天 2=偶尔 3=已戒 4=从不)"),
    "hypertension": (0, 1, "高血压(医生告知, 1=有)"),
    "diabetes": (0, 1, "糖尿病(医生告知, 1=有)"),
}

# 风险分档（括号内为留出测试集实测高胆固醇率，人群基线 31.56%）
RISK_BANDS = {
    "低风险": "实测约 9.1%（约为人群基线 31.6% 的 0.29 倍）",
    "人群基线": "实测约 27.7%（与人群平均持平）",
    "中等风险": "实测约 45.2%（约为基线 1.43 倍）",
    "较高风险": "实测约 62.9%（约为基线 1.99 倍）—— 建议检测血脂",
    "高风险": "实测约 77.6%（约为基线 2.46 倍）—— 建议检测血脂",
}
BAND_TOP = "高风险"
BAND_EDGES = []   # 边界由模型元数据中的 risk_band_edges 提供


# ---------------------------------------------------------------------------
def _load_metadata():
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_medians(metadata):
    prep = (metadata or {}).get("preprocessing", {}) or {}
    stats = prep.get("imputer_statistics")
    order = prep.get("feature_order", FEATURE_ORDER)
    if stats:
        return np.array([float(stats[c]) for c in order], dtype=float), order, \
            "metadata:imputer_statistics"
    try:
        import joblib
        path = os.path.join(MODELS_DIR, "imputer.joblib")
        if os.path.exists(path):
            imp = joblib.load(path)
            return np.asarray(imp.statistics_, dtype=float), list(order), "imputer.joblib"
    except ImportError:
        pass
    raise FileNotFoundError(
        f"错误: 未找到插补参数。请先运行 train_model.py。查找路径: {MODELS_DIR}")


def _load_model():
    import xgboost as xgb
    for name in ("hyperlipidemia_xgb_model.json", "hyperlipidemia_xgb_model.ubj"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            booster = xgb.Booster()
            booster.load_model(path)
            return booster, name
    raise FileNotFoundError(
        f"错误: 未找到模型工件。请先运行:\n"
        f"  python {os.path.join(SCRIPT_DIR, 'train_model.py')}")


def load_artifacts():
    metadata = _load_metadata()
    model, model_src = _load_model()
    medians, order, median_src = _load_medians(metadata)
    return model, medians, order, metadata, {"model": model_src, "imputer": median_src}


# ---------------------------------------------------------------------------
def impute_row(values, medians, order):
    row, imputed = [], []
    for i, name in enumerate(order):
        v = values.get(name)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            row.append(float(medians[i]))
            imputed.append(name)
        else:
            row.append(float(v))
    return np.array(row, dtype=float).reshape(1, -1), imputed


def _positive_proba(model, X, order):
    import xgboost as xgb
    if isinstance(model, xgb.Booster):
        dm = xgb.DMatrix(np.asarray(X, dtype=float), feature_names=list(order))
        return float(np.asarray(model.predict(dm)).reshape(-1)[0])
    return float(model.predict_proba(X)[0][1])


def risk_band(probability, metadata):
    edges = (metadata or {}).get("risk_band_edges") or []
    for edge, name in edges:
        if probability < edge:
            return name, RISK_BANDS.get(name, "")
    top = (metadata or {}).get("risk_band_top", BAND_TOP)
    return top, RISK_BANDS.get(top, "")


def predict_patient(model=None, medians=None, order=None, metadata=None,
                    threshold=None, **features):
    """预测个体高脂血症风险，返回结构化 dict。"""
    if model is None or medians is None:
        model, medians, order, metadata, sources = load_artifacts()
    else:
        order = order or FEATURE_ORDER
        sources = {"model": "injected", "imputer": "injected"}

    values = {name: features.get(name) for name in order}
    missing_core = [f for f in CORE_FEATURES if f in order and values.get(f) is None]
    if missing_core:
        raise ValueError(
            f"错误: 缺少核心指标 {missing_core}。\n"
            f"核心指标 {[f for f in CORE_FEATURES if f in order]}"
            f"（年龄/BMI/吸烟）不可用中位数替代，否则会低估风险。\n"
            f"强烈建议同时提供: {STRONGLY_RECOMMENDED}（模型的两个最强特征）")

    for name in order:
        v = values.get(name)
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            raise ValueError(f"错误: 特征 '{name}' 取值非法: {values[name]}")
        lo, hi, desc = FEATURE_LIMITS[name]
        if v < lo or v > hi:
            raise ValueError(
                f"错误: 特征 '{name}' ({desc}) = {values[name]} "
                f"超出合理范围 [{lo}, {hi}]")

    metadata = metadata or {}
    default_threshold = float(metadata.get("decision_threshold", 0.305))
    used_threshold = float(threshold) if threshold is not None else default_threshold

    X, imputed = impute_row(values, medians, order)
    probability = _positive_proba(model, X, order)
    prediction = int(probability >= used_threshold)
    level, note = risk_band(probability, metadata)

    # 若最强的两个特征被插补，明确提示风险被低估
    imputed_strong = [f for f in STRONGLY_RECOMMENDED if f in imputed]

    return {
        "probability": round(probability, 4),
        "probability_pct": f"{probability:.2%}",
        "prediction": prediction,
        "label": "高于判定阈值" if prediction == 1 else "低于判定阈值",
        "risk_level": level,
        "risk_note": note,
        "decision_threshold": used_threshold,
        "threshold_default": default_threshold,
        "threshold_note": "默认阈值由训练集交叉验证最大化平衡准确率选出，非固定 0.5",
        "features": {n: (None if values[n] is None else float(values[n])) for n in order},
        "imputed_features": imputed,
        "imputed_strong_features": imputed_strong,
        "underestimation_warning": (
            f"注意: {imputed_strong} 未提供，已按人群分布中位数插补（等价于「无该病」）。"
            f"这两项是本模型最重要的特征，缺失会**系统性低估**风险，"
            f"请补齐后重新评估。" if imputed_strong else None),
        "core_features": [f for f in CORE_FEATURES if f in order],
        "model": metadata.get("model_type", "xgboost.XGBClassifier"),
        "model_version": metadata.get("version"),
        "model_trained_at": metadata.get("trained_at"),
        "_sources": sources,
    }


def _print_report(result, sources):
    print("\n" + "=" * 74)
    print("            高脂血症（高胆固醇）风险预测结果")
    print("=" * 74)
    print(f"\n风险概率: {result['probability_pct']}   "
          f"(判定阈值 {result['decision_threshold']:.3f})")
    print(f"风险等级: {result['risk_level']}")
    print(f"参考    : {result['risk_note']}")
    print("\n输入特征:")
    for k, v in result["features"].items():
        print(f"  {k:13s} = {'未提供  [中位数插补]' if v is None else v}")
    if result["imputed_features"]:
        print(f"\n被插补的特征: {', '.join(result['imputed_features'])}")
    if result["underestimation_warning"]:
        print(f"\n⚠️  {result['underestimation_warning']}")
    print(f"\n模型工件: {sources['model']}  |  插补参数: {sources['imputer']}")
    print("=" * 74)
    print("⚠️  本模型**不含任何血脂检验值**（无总胆固醇、LDL-C、甘油三酯）。")
    print("    原因是这些指标本身即高脂血症的诊断依据，用其预测近乎循环论证。")
    print("    本模型仅用于【风险分层】，确诊必须依靠血脂检测，请遵医嘱。")
    print("    另注：高脂血症无症状，「曾被医生告知」的前提是先做过血脂检测，")
    print("    故本模型部分反映的是【检测机会 / 医疗可及性】，而非纯生物学风险。")


def main():
    p = argparse.ArgumentParser(description="高脂血症风险预测（单个体）")
    p.add_argument("--age", type=float, required=True, help="年龄(18-85)")
    p.add_argument("--bmi", type=float, required=True, help="BMI体重指数")
    p.add_argument("--smoker", type=int, required=True,
                   help="吸烟状态(1=每天 2=偶尔 3=已戒 4=从不)")
    p.add_argument("--height", type=float, default=None, dest="height_cm", help="身高(cm)")
    p.add_argument("--weight", type=float, default=None, dest="weight_kg", help="体重(kg)")
    p.add_argument("--hypertension", type=int, default=None,
                   help="高血压(1=有 0=无)  ← 强烈建议提供")
    p.add_argument("--diabetes", type=int, default=None,
                   help="糖尿病(1=有 0=无)  ← 强烈建议提供")
    p.add_argument("--threshold", type=float, default=None, help="自定义判定阈值")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = p.parse_args()

    try:
        result = predict_patient(
            age=args.age, bmi=args.bmi, smoker=args.smoker,
            height_cm=args.height_cm, weight_kg=args.weight_kg,
            hypertension=args.hypertension, diabetes=args.diabetes,
            threshold=args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result, result["_sources"])


if __name__ == "__main__":
    main()
