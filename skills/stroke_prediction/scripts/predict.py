#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑卒中（Stroke）风险预测 - 单个体推理脚本
================================================================
加载 models/ 下的推理工件，对个体做脑卒中风险二分类预测。

预处理与训练一致：缺失特征用训练中位数插补，XGBoost 输出概率，
再用**训练集交叉验证选定的阈值**（默认 0.04，不是 0.5）判定。

注意: 本任务阳性率仅约 4.2%，概率数值本身偏小属正常现象。
      风险等级对照的是实测患病率，不是 30%/70% 这种直觉分档。

用法:
    # 6 项主要临床指标即可预测
    python scripts/predict.py --age 68 --bmi 28.0 --smoker 3 --diabetes 1 \
        --hypertension 1 --high-chol 1

    # 补充更多指标（可选）
    python scripts/predict.py --age 68 --bmi 28.0 --smoker 3 --diabetes 1 \
        --hypertension 1 --high-chol 1 --gen-health 3 --copd 0 --json

作为模块调用:
    from predict import predict_patient
    predict_patient(age=68, bmi=28.0, smoker=3, diabetes=1, hypertension=1, high_chol=1)
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

FEATURE_ORDER = [
    "age", "bmi", "height_cm", "weight_kg", "smoker", "diabetes",
    "hypertension", "high_chol", "gen_health", "exercise", "education",
    "income", "drinks_wk", "depression", "copd",
]

# 主要临床风险因素：必须显式提供，避免被中位数悄悄"默认成健康状态"
CORE_FEATURES = ["age", "bmi", "smoker", "diabetes", "hypertension", "high_chol"]
OPTIONAL_FEATURES = [f for f in FEATURE_ORDER if f not in CORE_FEATURES]

FEATURE_LIMITS = {
    "age": (18, 80, "年龄(岁)"),
    "bmi": (10.0, 90.0, "BMI体重指数(kg/m^2)"),
    "height_cm": (90.0, 240.0, "身高(cm)"),
    "weight_kg": (20.0, 300.0, "体重(kg)"),
    "smoker": (1, 4, "吸烟状态(1=每天 2=偶尔 3=已戒 4=从不)"),
    "diabetes": (0, 1, "糖尿病(1=有)"),
    "hypertension": (0, 1, "高血压(1=有)"),
    "high_chol": (0, 1, "高胆固醇(1=有)"),
    "gen_health": (1, 5, "自评健康(1=优 2=良 3=一般 4=差 5=很差)"),
    "exercise": (0, 1, "休闲运动(1=有)"),
    "education": (1, 6, "教育程度(1-6)"),
    "income": (1, 8, "家庭年收入档(1-8)"),
    "drinks_wk": (0, 200, "每周饮酒量(杯)"),
    "depression": (0, 1, "抑郁障碍(1=有)"),
    "copd": (0, 1, "慢阻肺/肺气肿(1=有)"),
}

# 风险分档：括号内为留出测试集上该档的实测脑卒中患病率（模型校准良好）
RISK_BANDS = [
    (0.04, "低风险", "实测患病率约 0.6%–2.8%；保持健康生活方式，定期体检"),
    (0.08, "中等风险", "实测患病率约 6.1%；建议控制血压血糖、戒烟、定期复查"),
    (0.15, "较高风险", "实测患病率约 11.0%；建议尽快就医，评估血压、血糖、血脂与颈动脉"),
    (0.30, "高风险", "实测患病率约 19.1%；建议进行脑血管专项评估（含心律与颈动脉超声）"),
    (1.01, "极高风险", "实测患病率约 35.4%；强烈建议尽快就医做神经内科/卒中专科评估"),
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

    for name in ("stroke_xgb_model.json", "stroke_xgb_model.ubj"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            booster = xgb.Booster()
            booster.load_model(path)
            return booster, name
    raise FileNotFoundError(
        f"错误: 未找到模型工件。请先运行:\n  python {os.path.join(SCRIPT_DIR, 'train_model.py')}")


def load_artifacts():
    metadata = _load_metadata()
    model, model_src = _load_model()
    medians, order, median_src = _load_medians(metadata)
    return model, medians, order, metadata, {"model": model_src, "imputer": median_src}


# ---------------------------------------------------------------------------
# 预处理与推理
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


def risk_band(probability):
    for threshold, level, note in RISK_BANDS:
        if probability < threshold:
            return level, note
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict_patient(model=None, medians=None, order=None, metadata=None, **features):
    """预测个体脑卒中风险，返回结构化 dict。"""
    if model is None or medians is None:
        model, medians, order, metadata, sources = load_artifacts()
    else:
        order = order or FEATURE_ORDER
        sources = {"model": "injected", "imputer": "injected"}

    values = {name: features.get(name) for name in order}
    core = [f for f in CORE_FEATURES if f in order]
    missing_core = [f for f in core
                    if values.get(f) is None
                    or (isinstance(values.get(f), float) and not np.isfinite(values.get(f)))]
    if missing_core:
        raise ValueError(
            f"错误: 缺少核心指标 {missing_core}。\n"
            f"本模型要求显式提供主要风险因素 {core}，\n"
            f"因为它们不能被中位数替代（否则等价于假设该风险因素不存在，会低估风险）。\n"
            f"可省略的次要指标: {[f for f in order if f not in core]}"
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
    threshold = float(metadata.get("decision_threshold", 0.04))

    X, imputed = impute_row(values, medians, order)
    probability = _positive_proba(model, X, order)
    prediction = int(probability >= threshold)
    level, note = risk_band(probability)

    return {
        "probability": round(probability, 4),
        "probability_pct": f"{probability:.2%}",
        "prediction": prediction,
        "label": "脑卒中阳性" if prediction == 1 else "脑卒中阴性",
        "risk_level": level,
        "risk_note": note,
        "decision_threshold": threshold,
        "threshold_note": "阈值由训练集交叉验证选定（最大化平衡准确率），非固定 0.5",
        "features": {n: (None if values[n] is None else float(values[n])) for n in order},
        "imputed_features": imputed,
        "core_features": core,
        "model": metadata.get("model_type", "xgboost.XGBClassifier"),
        "model_version": metadata.get("version"),
        "model_trained_at": metadata.get("trained_at"),
        "_sources": sources,
    }


def _print_report(result, sources):
    print("\n" + "=" * 66)
    print("            脑卒中（Stroke）风险预测结果")
    print("=" * 66)
    print(f"\n风险概率: {result['probability_pct']}   (判定阈值 {result['decision_threshold']})")
    print(f"预测结果: {'阳性 - 提示存在脑卒中' if result['prediction'] == 1 else '阴性 - 未提示脑卒中'}")
    print(f"风险等级: {result['risk_level']}")
    print(f"参考    : {result['risk_note']}")
    print("\n输入特征:")
    for k, v in result["features"].items():
        if v is None:
            print(f"  {k:13s} = 未提供  [中位数插补]")
        else:
            print(f"  {k:13s} = {v}")
    if result["imputed_features"]:
        print(f"\n被插补的次要指标: {', '.join(result['imputed_features'])}")
    print(f"\n模型工件: {sources['model']}  |  插补参数: {sources['imputer']}")
    print("=" * 66)
    print("提示: 本模型基于美国 BRFSS 人群调查数据，仅供健康风险筛查参考，")
    print("      不能替代专业医生的诊断。")


def main():
    p = argparse.ArgumentParser(description="脑卒中（Stroke）风险预测（单个体）")
    p.add_argument("--age", type=float, required=True, help="年龄(18-80)")
    p.add_argument("--bmi", type=float, required=True, help="BMI体重指数")
    p.add_argument("--smoker", type=int, required=True,
                   help="吸烟状态(1=每天 2=偶尔 3=已戒 4=从不)")
    p.add_argument("--diabetes", type=int, required=True, help="糖尿病(1=有 0=无)")
    p.add_argument("--hypertension", type=int, required=True, help="高血压(1=有 0=无)")
    p.add_argument("--high-chol", type=int, required=True, dest="high_chol",
                   help="高胆固醇(1=有 0=无)")
    p.add_argument("--height", type=float, default=None, help="身高(cm)")
    p.add_argument("--weight", type=float, default=None, help="体重(kg)")
    p.add_argument("--gen-health", type=int, default=None, dest="gen_health",
                   help="自评健康(1=优 2=良 3=一般 4=差 5=很差)")
    p.add_argument("--exercise", type=int, default=None, help="休闲运动(1=有 0=无)")
    p.add_argument("--education", type=int, default=None, help="教育程度(1-6)")
    p.add_argument("--income", type=int, default=None, help="家庭年收入档(1-8)")
    p.add_argument("--drinks-wk", type=float, default=None, dest="drinks_wk",
                   help="每周饮酒量(杯)")
    p.add_argument("--depression", type=int, default=None, help="抑郁障碍(1=有 0=无)")
    p.add_argument("--copd", type=int, default=None, help="慢阻肺/肺气肿(1=有 0=无)")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = p.parse_args()

    try:
        result = predict_patient(
            age=args.age, bmi=args.bmi, smoker=args.smoker, diabetes=args.diabetes,
            hypertension=args.hypertension, high_chol=args.high_chol,
            height_cm=args.height, weight_kg=args.weight, gen_health=args.gen_health,
            exercise=args.exercise, education=args.education, income=args.income,
            drinks_wk=args.drinks_wk, depression=args.depression, copd=args.copd,
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
