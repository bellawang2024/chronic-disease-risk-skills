#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冠心病（CHD）风险预测模型 - 训练与验证脚本
================================================================
数据来源: BRFSS 2017 (CDC Behavioral Risk Factor Surveillance System)
          美国 50 州 + 属地的全人群电话调查抽样 —— 全人群, 不限性别/种族
样本:     445,872 名成人 (>=18 岁)
特征:     15 个简洁的风险因素指标 (问卷 + 派生)
标签:     _MICHD — CDC 官方派生变量 "曾被医生告知患有冠心病或心肌梗死"
模型:     XGBoost 二分类

重要: 本任务阳性率仅约 8.8%，**准确率(accuracy)不是有效指标**
      (多数类基线即有 91.2%)。评估以 AUC + 平衡准确率 + 敏感度/特异度为准，
      决策阈值由训练集交叉验证选出，而非固定 0.5。
================================================================
用法:
    python scripts/train_model.py
    python scripts/train_model.py --quick
================================================================
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SKILL_DIR, "data")
MODELS_DIR = os.path.join(SKILL_DIR, "models")
RESULTS_DIR = os.path.join(SKILL_DIR, "results")
DEFAULT_DATA = os.path.join(DATA_DIR, "coronary_heart_disease_brfss.csv")

# ---------------------------------------------------------------------------
# 特征定义
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "age", "bmi", "height_cm", "weight_kg", "smoker", "diabetes",
    "hypertension", "high_chol", "gen_health", "exercise", "education",
    "income", "drinks_wk", "depression", "copd",
]
# 说明: 派生数据集在构建阶段就完全未提取 SEX / _RACE，
# 因此模型在结构上不可能使用性别、种族等任何人群划分属性。
TARGET = "label"

FEATURE_NAMES_CN = {
    "age": "年龄",
    "bmi": "BMI体重指数",
    "height_cm": "身高",
    "weight_kg": "体重",
    "smoker": "吸烟状态(1=每天 2=偶尔 3=已戒 4=从不)",
    "diabetes": "糖尿病",
    "hypertension": "高血压(医生告知)",
    "high_chol": "高胆固醇(医生告知)",
    "gen_health": "自评健康状况(1=优 2=良 3=一般 4=差 5=很差)",
    "exercise": "过去30天是否有休闲运动",
    "education": "教育程度(1-6)",
    "income": "家庭年收入档(1-8)",
    "drinks_wk": "每周饮酒量(杯)",
    "depression": "抑郁障碍(医生告知)",
    "copd": "慢性阻塞性肺病/肺气肿",
}
FEATURE_UNITS = {
    "age": "岁", "bmi": "kg/m^2", "height_cm": "cm", "weight_kg": "kg",
    "smoker": "编码1-4", "diabetes": "1/0", "hypertension": "1/0",
    "high_chol": "1/0", "gen_health": "编码1-5", "exercise": "1/0",
    "education": "编码1-6", "income": "编码1-8", "drinks_wk": "杯/周",
    "depression": "1/0", "copd": "1/0",
}
FEATURE_RANGES = {
    "age": (18, 80), "bmi": (10.0, 90.0), "height_cm": (90.0, 240.0),
    "weight_kg": (20.0, 300.0), "smoker": (1, 4), "diabetes": (0, 1),
    "hypertension": (0, 1), "high_chol": (0, 1), "gen_health": (1, 5),
    "exercise": (0, 1), "education": (1, 6), "income": (1, 8),
    "drinks_wk": (0, 200), "depression": (0, 1), "copd": (0, 1),
}

# ---------------------------------------------------------------------------
# 训练配置（固定，保证可复现）
# ---------------------------------------------------------------------------
SPLIT_CONFIG = {"test_size": 0.2, "random_state": 42, "stratify": True}
CV_CONFIG = {"n_splits": 5, "n_repeats": 3, "random_state": 42}
THRESHOLD_GRID = (0.02, 0.80, 0.01)

# 由训练集内层 5 折 GridSearchCV 按 AUC 选出的超参数
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 10,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

# 验收标准：本任务用 AUC 与平衡准确率，不用 plain accuracy（见文件头说明）
ACCEPTANCE = {
    "holdout_auc": 0.80,
    "holdout_balanced_accuracy": 0.70,
    "cv_auc": 0.80,
    "cv_balanced_accuracy": 0.70,
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_dataframe(data_path=None):
    path = data_path or DEFAULT_DATA
    if not os.path.exists(path):
        raise SystemExit(
            f"错误: 未找到数据文件: {path}\n"
            f"请运行 python {os.path.join(SCRIPT_DIR, 'fetch_data.py')} 重建数据集。"
        )
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"错误: 数据缺少列: {missing}")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return df, {"path": os.path.abspath(path), "sha256": sha, "n_samples": int(len(df))}


def feature_list():
    """15 个风险因素特征；性别/种族在数据构建阶段即未提取。"""
    return list(ALL_FEATURES)


def make_pipeline():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", xgb.XGBClassifier(**XGB_PARAMS))])


# ---------------------------------------------------------------------------
# 阈值选择
# ---------------------------------------------------------------------------
def tune_threshold(y_true, proba):
    """在给定（训练）概率上选使平衡准确率最大的阈值。不接触测试集。"""
    ths = np.arange(*THRESHOLD_GRID)
    scores = [balanced_accuracy_score(y_true, (proba >= t).astype(int)) for t in ths]
    i = int(np.argmax(scores))
    return float(round(ths[i], 2)), float(scores[i])


def binary_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "pr_auc_baseline": round(float(np.mean(y_true)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "sensitivity": round(float(recall_score(y_true, pred)), 4),
        "specificity": round(float(recall_score(y_true, pred, pos_label=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "plain_accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "majority_baseline_accuracy": round(float(1 - np.mean(y_true)), 4),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate_holdout(df, features, verbose=True):
    X, y = df[features], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=SPLIT_CONFIG["test_size"],
        random_state=SPLIT_CONFIG["random_state"],
        stratify=y if SPLIT_CONFIG["stratify"] else None,
    )

    # 阈值仅在训练集上通过交叉验证选定
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_proba = cross_val_predict(make_pipeline(), X_tr, y_tr, cv=skf,
                                    method="predict_proba", n_jobs=-1)[:, 1]
    threshold, train_ba = tune_threshold(y_tr, train_proba)

    model = make_pipeline().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = binary_metrics(y_te, proba, threshold)
    metrics.update({"n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                    "threshold_from": "train_cv", "train_cv_balanced_accuracy": round(train_ba, 4)})

    if verbose:
        print(f"\n----- 留出测试集评估（{len(X_tr):,} 训练 / {len(X_te):,} 测试）-----")
        print(f"阈值由训练集 CV 选定: {threshold:.2f} (训练集 CV 平衡准确率 {train_ba:.4f})")
        print(f"AUC:           {metrics['auc']:.4f}")
        print(f"PR-AUC:        {metrics['pr_auc']:.4f}  (基线 {metrics['pr_auc_baseline']:.4f})")
        print(f"平衡准确率:     {metrics['balanced_accuracy']:.4f}")
        print(f"敏感度/特异度:  {metrics['sensitivity']:.4f} / {metrics['specificity']:.4f}")
        print(f"精确率/F1:     {metrics['precision']:.4f} / {metrics['f1']:.4f}")
        print(f"普通准确率:     {metrics['plain_accuracy']:.4f}  "
              f"(多数类基线 {metrics['majority_baseline_accuracy']:.4f} — 本任务该指标无参考性)")
        cm = metrics["confusion_matrix"]
        print(f"混淆矩阵: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    return metrics


def evaluate_cv(df, features, quick=False, verbose=True):
    """重复交叉验证：每折内部独立选阈值，避免阈值信息泄漏。"""
    X, y = df[features], df[TARGET]
    n_repeats = 1 if quick else CV_CONFIG["n_repeats"]
    cv = RepeatedStratifiedKFold(n_splits=CV_CONFIG["n_splits"], n_repeats=n_repeats,
                                 random_state=CV_CONFIG["random_state"])
    keys = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "threshold"]
    res = {k: [] for k in keys}

    total = CV_CONFIG["n_splits"] * n_repeats
    for i, (tr, va) in enumerate(cv.split(X, y)):
        mdl = make_pipeline().fit(X.iloc[tr], y.iloc[tr])
        t, _ = tune_threshold(y.iloc[tr], mdl.predict_proba(X.iloc[tr])[:, 1])
        pv = mdl.predict_proba(X.iloc[va])[:, 1]
        pv_ = (pv >= t).astype(int)
        res["auc"].append(roc_auc_score(y.iloc[va], pv))
        res["pr_auc"].append(average_precision_score(y.iloc[va], pv))
        res["balanced_accuracy"].append(balanced_accuracy_score(y.iloc[va], pv_))
        res["sensitivity"].append(recall_score(y.iloc[va], pv_))
        res["specificity"].append(recall_score(y.iloc[va], pv_, pos_label=0))
        res["f1"].append(f1_score(y.iloc[va], pv_))
        res["threshold"].append(t)
        if verbose:
            print(f"  fold {i+1:2d}/{total}  AUC {res['auc'][-1]:.4f}  "
                  f"平衡准确率 {res['balanced_accuracy'][-1]:.4f}  阈值 {t:.2f}")

    metrics = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
               for k, v in res.items()}
    metrics.update({"n_splits": CV_CONFIG["n_splits"], "n_repeats": n_repeats,
                    "n_folds_total": total,
                    "note": "每折内部自行选择阈值，阈值信息未泄漏到验证折"})
    if verbose:
        print(f"\n----- {n_repeats}×{CV_CONFIG['n_splits']} 折重复交叉验证（全量 {len(X):,} 样本）-----")
        for k in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            print(f"{k:20s}: {metrics[k]['mean']:.4f}  ± {metrics[k]['std']:.4f}")
    return metrics


def check_acceptance(holdout, cv, verbose=True):
    checks = {
        "holdout_auc": holdout["auc"] >= ACCEPTANCE["holdout_auc"],
        "holdout_balanced_accuracy":
            holdout["balanced_accuracy"] >= ACCEPTANCE["holdout_balanced_accuracy"],
        "cv_auc": cv["auc"]["mean"] >= ACCEPTANCE["cv_auc"],
        "cv_balanced_accuracy":
            cv["balanced_accuracy"]["mean"] >= ACCEPTANCE["cv_balanced_accuracy"],
    }
    if verbose:
        labels = {
            "holdout_auc": f"留出测试集 AUC        >= {ACCEPTANCE['holdout_auc']:.2f}",
            "holdout_balanced_accuracy":
                f"留出测试集 平衡准确率 >= {ACCEPTANCE['holdout_balanced_accuracy']:.2f}",
            "cv_auc": f"交叉验证   AUC        >= {ACCEPTANCE['cv_auc']:.2f}",
            "cv_balanced_accuracy":
                f"交叉验证   平衡准确率 >= {ACCEPTANCE['cv_balanced_accuracy']:.2f}",
        }
        print("\n----- 验收标准检查 -----")
        print("  （本任务阳性率仅约 8.8%，故不用普通准确率作为验收指标）")
        for k, ok in checks.items():
            print(f"  [{'通过' if ok else '未通过'}] {labels[k]}")
        print(f"\n  综合结论: {'全部达标' if all(checks.values()) else '未达标'}")
    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def train_model(data_path=None, quick=False, verbose=True):
    df, data_info = load_dataframe(data_path)
    features = feature_list()
    y = df[TARGET]

    if verbose:
        n_pos = int(y.sum())
        print("=" * 72)
        print(" 冠心病（CHD）风险预测模型 - 训练与验证")
        print("=" * 72)
        print(f"数据文件: {data_info['path']}")
        print(f"SHA256  : {data_info['sha256']}")
        print(f"样本数  : {data_info['n_samples']:,}   特征数: {len(features)}")
        print(f"类别分布: 阳性(冠心病/心梗) {n_pos:,} / 阴性 {len(df)-n_pos:,}  "
              f"-> 患病率 {y.mean():.2%}")
        print(f"多数类基线准确率: {1-y.mean():.4f}  <- 准确率在本任务上不具参考性")
        print("人群划分属性: 未纳入（性别/种族在数据构建阶段即未提取）")

    holdout = evaluate_holdout(df, features, verbose=verbose)
    cv = evaluate_cv(df, features, quick=quick, verbose=verbose)
    acceptance = check_acceptance(holdout, cv, verbose=verbose)

    # 用全量数据重训发布模型；阈值沿用留出评估阶段由训练集 CV 选定的值
    threshold = holdout["threshold"]
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(df[features])
    model = xgb.XGBClassifier(**XGB_PARAMS).fit(Xt, y)
    if verbose:
        print(f"\n----- 发布模型（用全量 {len(df):,} 条样本重训，阈值 {threshold:.2f}）-----")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_json = os.path.join(MODELS_DIR, "chd_xgb_model.json")
    model_ubj = os.path.join(MODELS_DIR, "chd_xgb_model.ubj")
    model.save_model(model_json)
    model.save_model(model_ubj)
    try:
        import joblib

        joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))
    except ImportError:
        pass

    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    importance["feature_cn"] = importance["feature"].map(FEATURE_NAMES_CN)
    importance["importance"] = importance["importance"].round(4)

    if verbose:
        print("\n----- 影响冠心病的关键特征（按重要性排序）-----")
        for _, r in importance.iterrows():
            print(f"  {r['feature']:12s} ({r['feature_cn'][:22]}): {r['importance']:.4f}")

    medians = {f: round(float(v), 4) for f, v in zip(features, imputer.statistics_)}
    n_pos = int(y.sum())

    metadata = {
        "skill_name": "chd_prediction",
        "version": "1.0",
        "model_type": "xgboost.XGBClassifier",
        "task": "binary_classification",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "full_dataset",
        "dataset": {
            "name": "BRFSS 2017 (Behavioral Risk Factor Surveillance System)",
            "owner": "U.S. Centers for Disease Control and Prevention (CDC)",
            "survey_year": 2017,
            "target_population": "美国 50 州及属地的成年居民（全人群电话调查抽样，不限性别、种族）",
            "cohort_definition": "≥18 岁且有冠心病/心梗作答记录的受访者",
            "file": "data/coronary_heart_disease_brfss.csv",
            "sha256": data_info["sha256"],
            "n_samples": data_info["n_samples"],
            "n_features": len(features),
            "features": features,
            "features_cn": FEATURE_NAMES_CN,
            "feature_units": FEATURE_UNITS,
            "sensitive_features_included": False,
            "sensitive_features_note": "派生数据集未提取 SEX / _RACE，模型结构上无法使用人群划分属性",
            "target": TARGET,
            "target_definition": "BRFSS 派生变量 _MICHD = 曾被医生告知患有冠心病(CHD)或心肌梗死(MI)",
            "target_labels": {"0": "无冠心病/心梗", "1": "有冠心病/心梗"},
            "positive_count": n_pos,
            "prevalence": round(float(y.mean()), 4),
            "n_respondents_raw": 450016,
        },
        "preprocessing": {
            "imputer": "sklearn.impute.SimpleImputer(strategy='median')",
            "imputer_statistics": medians,
            "feature_order": features,
            "scaling": "无（XGBoost 对特征尺度不敏感）",
        },
        "split": {**SPLIT_CONFIG, "n_train": holdout["n_train"], "n_test": holdout["n_test"]},
        "cross_validation": CV_CONFIG,
        "hyperparameters": XGB_PARAMS,
        "decision_threshold": threshold,
        "decision_threshold_note": (
            "阈值由训练集交叉验证最大化平衡准确率选出，未使用测试集；"
            "该任务阳性率仅约 8.8%，固定 0.5 会导致敏感度不足 0.08"
        ),
        "validation": {
            "holdout": holdout,
            "cross_validation": cv,
            "acceptance_criteria": ACCEPTANCE,
            "acceptance_passed": acceptance,
        },
        "artifacts": {
            "model_json": "chd_xgb_model.json",
            "model_ubj": "chd_xgb_model.ubj",
            "imputer": "imputer.joblib",
        },
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "holdout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(holdout, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    importance.to_json(os.path.join(RESULTS_DIR, "feature_importance.json"),
                       orient="records", force_ascii=False, indent=2)

    if verbose:
        print("\n----- 推理工件已保存 -----")
        print(f"  模型 (json) : {model_json}")
        print(f"  模型 (ubj)  : {model_ubj}")
        print(f"  元数据      : {os.path.join(MODELS_DIR, 'model_metadata.json')}")
        print(f"  决策阈值    : {threshold:.2f}")
        print("\n训练完成!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="冠心病风险预测模型 - 训练与验证")
    parser.add_argument("--data", default=None, help="外部训练 CSV")
    parser.add_argument("--quick", action="store_true", help="减少交叉验证重复次数")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    meta = train_model(data_path=args.data, quick=args.quick, verbose=not args.quiet)
    if not all(meta["validation"]["acceptance_passed"].values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
