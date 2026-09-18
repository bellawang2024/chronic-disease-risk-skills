#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高血压风险预测 - 单患者推理脚本
================================================================
加载 models/ 下的推理工件，对个体做高血压风险二分类预测。

预处理与训练完全一致：缺失特征用训练中位数插补，然后 XGBoost 预测，
阈值 0.5 判定（无标准化，XGBoost 对尺度不敏感）。

用法:
    # 只需 4 个核心指标即可预测，其余自动用训练中位数插补
    python scripts/predict.py --age 55 --sex male --bmi 27.5 --waist 95

    # 提供更多指标可获得更准确的评估
    python scripts/predict.py --age 55 --sex male --bmi 27.5 --waist 95 \
        --hba1c 6.1 --cholesterol 210 --hdl 42 --creatinine 1.0 --uric-acid 6.8 \
        --smoker 1 --diabetes 0 --pulse 78 --height 172 --income-pir 2.5 --json

作为模块调用:
    from predict import predict_patient
    predict_patient(age=55, sex_male=1, bmi=27.5, waist_cm=95)
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

# 特征顺序（与 model_metadata.json 中 preprocessing.feature_order 一致）
FEATURE_ORDER = [
    "age", "sex_male", "bmi", "waist_cm", "height_cm", "race", "education",
    "income_pir", "smoker", "diabetes", "pulse", "hba1c", "cholesterol",
    "hdl", "creatinine", "uric_acid",
]

# 必须有值才能给出有意义预测的核心指标（sex_male 仅当模型使用该特征时才必需）
CORE_FEATURES = ["age", "sex_male", "bmi", "waist_cm"]

# 未提供时用训练中位数插补的可选指标
OPTIONAL_FEATURES = [f for f in FEATURE_ORDER if f not in CORE_FEATURES]

# 合理取值范围（依据 NHANES 实际分布留出余量）
FEATURE_LIMITS = {
    "age": (18, 80, "年龄(岁)"),
    "sex_male": (0, 1, "性别(1=男,0=女)"),
    "bmi": (10.0, 90.0, "BMI体重指数(kg/m^2)"),
    "waist_cm": (40.0, 200.0, "腰围(cm)"),
    "height_cm": (120.0, 220.0, "身高(cm)"),
    "race": (1, 6, "种族族裔编码"),
    "education": (1, 5, "教育程度编码"),
    "income_pir": (0.0, 5.0, "家庭收入贫困比"),
    "smoker": (0, 1, "吸烟(1=一生≥100支)"),
    "diabetes": (0, 1, "糖尿病(1=是)"),
    "pulse": (20, 200, "静息脉搏(次/分)"),
    "hba1c": (3.0, 20.0, "糖化血红蛋白(%)"),
    "cholesterol": (50.0, 600.0, "总胆固醇(mg/dL)"),
    "hdl": (5.0, 200.0, "HDL(mg/dL)"),
    "creatinine": (0.1, 20.0, "血清肌酐(mg/dL)"),
    "uric_acid": (0.5, 20.0, "血清尿酸(mg/dL)"),
}

RISK_BANDS = [
    (0.30, "低风险", "保持健康生活方式，定期监测血压"),
    (0.70, "中等风险", "建议控制体重与腰围、减少钠盐摄入、增加运动，并定期测量血压"),
    (1.01, "高风险", "建议尽快就医，进行规范的血压测量与心血管风险评估"),
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
    """优先用元数据中记录的中位数（无需 joblib），否则退回 joblib 工件。"""
    stats = ((metadata or {}).get("preprocessing", {}) or {}).get("imputer_statistics")
    order = ((metadata or {}).get("preprocessing", {}) or {}).get("feature_order", FEATURE_ORDER)
    if stats:
        medians = np.array([float(stats[c]) for c in order], dtype=float)
        return medians, order, "metadata:imputer_statistics"

    try:
        import joblib

        path = os.path.join(MODELS_DIR, "imputer.joblib")
        if os.path.exists(path):
            imputer = joblib.load(path)
            return np.asarray(imputer.statistics_, dtype=float), list(order), "imputer.joblib"
    except ImportError:
        pass

    raise FileNotFoundError(
        f"错误: 未找到插补参数。请先运行 train_model.py。查找路径: {MODELS_DIR}"
    )


def _load_model():
    import xgboost as xgb

    for name in ("hypertension_xgb_model.json", "hypertension_xgb_model.ubj"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            booster = xgb.Booster()
            booster.load_model(path)
            return booster, name

    raise FileNotFoundError(
        f"错误: 未找到模型工件。请先运行:\n  python {os.path.join(SCRIPT_DIR, 'train_model.py')}"
    )


def load_artifacts():
    """返回 (model, medians, feature_order, metadata, sources)。"""
    metadata = _load_metadata()
    model, model_src = _load_model()
    medians, order, median_src = _load_medians(metadata)
    return model, medians, order, metadata, {"model": model_src, "imputer": median_src}


# ---------------------------------------------------------------------------
# 预处理与推理
# ---------------------------------------------------------------------------
def impute_row(values, medians, order):
    """缺失(None/NaN)的特征用中位数填充，返回与训练一致的二维数组与插补清单。"""
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


def risk_band(probability):
    for threshold, level, advice in RISK_BANDS:
        if probability < threshold:
            return level, advice
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict_patient(model=None, medians=None, order=None, metadata=None, **features):
    """预测个体高血压风险。返回结构化 dict。

    核心指标 age/sex_male/bmi/waist_cm 必须提供；
    其余指标可省略，将用训练中位数插补并在 imputed_features 中标注。
    """
    if model is None or medians is None:
        model, medians, order, metadata, sources = load_artifacts()
    else:
        order = order or FEATURE_ORDER
        sources = {"model": "injected", "imputer": "injected"}

    # 对齐到模型实际使用的特征顺序
    values = {name: features.get(name) for name in order}

    # 核心指标必填（仅针对模型实际使用的特征）
    core = [f for f in CORE_FEATURES if f in order]
    missing_core = [f for f in core
                    if values.get(f) is None
                    or (isinstance(values.get(f), float) and not np.isfinite(values.get(f)))]
    if missing_core:
        raise ValueError(
            f"错误: 缺少核心指标 {missing_core}。"
            f"本模型的核心指标为 {core}，其余指标可省略（将用中位数插补）。"
        )

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
                f"错误: 特征 '{name}' ({desc}) = {values[name]} 超出合理范围 [{lo}, {hi}]")

    metadata = metadata or {}
    threshold = float(metadata.get("decision_threshold", 0.5))

    X, imputed = impute_row(values, medians, order)
    probability = _positive_proba(model, X, order)
    prediction = int(probability >= threshold)
    level, advice = risk_band(probability)

    return {
        "probability": round(probability, 4),
        "probability_pct": f"{probability:.2%}",
        "prediction": prediction,
        "label": "高血压" if prediction == 1 else "无高血压",
        "risk_level": level,
        "advice": advice,
        "decision_threshold": threshold,
        "features": {n: (None if values[n] is None else float(values[n])) for n in order},
        "imputed_features": imputed,
        "core_features": core,
        "model": metadata.get("model_type", "xgboost.XGBClassifier"),
        "model_version": metadata.get("version"),
        "model_trained_at": metadata.get("trained_at"),
        "_sources": sources,
    }


def _print_report(result, sources):
    print("\n" + "=" * 62)
    print("                高血压风险预测结果")
    print("=" * 62)
    print(f"\n风险概率: {result['probability_pct']}   (判定阈值 {result['decision_threshold']})")
    print(f"预测结果: {'阳性 - 高血压' if result['prediction'] == 1 else '阴性 - 无高血压'}")
    print(f"风险等级: {result['risk_level']}")
    print(f"建议: {result['advice']}")
    print("\n输入特征:")
    for k, v in result["features"].items():
        if v is None:
            print(f"  {k:12s} = 未提供  [中位数插补]")
        else:
            print(f"  {k:12s} = {v}")
    print(f"\n模型工件: {sources['model']}  |  插补参数: {sources['imputer']}")
    print("=" * 62)
    print("提示: 本结果仅供健康风险筛查参考，不能替代专业医生诊断。")


def main():
    p = argparse.ArgumentParser(description="高血压风险预测（单个体）")
    p.add_argument("--age", type=float, required=True, help="年龄(岁)")
    p.add_argument("--sex", choices=["male", "female"], default=None,
                   help="性别（仅当模型使用性别特征时必需）")
    p.add_argument("--bmi", type=float, required=True, help="BMI体重指数")
    p.add_argument("--waist", type=float, required=True, help="腰围(cm)")
    p.add_argument("--height", type=float, default=None, help="身高(cm)")
    p.add_argument("--race", type=int, default=None,
                   help="种族族裔(1=墨西哥裔 2=其他西语裔 3=非西语裔白人 4=非西语裔黑人 6=非西语裔亚裔)")
    p.add_argument("--education", type=int, default=None, help="教育程度(1-5)")
    p.add_argument("--income-pir", type=float, default=None, help="家庭收入贫困比(0-5)")
    p.add_argument("--smoker", type=int, default=None, help="吸烟(1=一生≥100支, 0=否)")
    p.add_argument("--diabetes", type=int, default=None, help="糖尿病(1=是, 0=否)")
    p.add_argument("--pulse", type=float, default=None, help="静息脉搏(次/分)")
    p.add_argument("--hba1c", type=float, default=None, help="糖化血红蛋白(%)")
    p.add_argument("--cholesterol", type=float, default=None, help="总胆固醇(mg/dL)")
    p.add_argument("--hdl", type=float, default=None, help="HDL(mg/dL)")
    p.add_argument("--creatinine", type=float, default=None, help="血清肌酐(mg/dL)")
    p.add_argument("--uric-acid", type=float, default=None, help="血清尿酸(mg/dL)")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = p.parse_args()

    try:
        result = predict_patient(
            age=args.age,
            sex_male=None if args.sex is None else (1 if args.sex == "male" else 0),
            bmi=args.bmi,
            waist_cm=args.waist,
            height_cm=args.height,
            race=args.race,
            education=args.education,
            income_pir=args.income_pir,
            smoker=args.smoker,
            diabetes=args.diabetes,
            pulse=args.pulse,
            hba1c=args.hba1c,
            cholesterol=args.cholesterol,
            hdl=args.hdl,
            creatinine=args.creatinine,
            uric_acid=args.uric_acid,
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
