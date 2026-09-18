#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病风险预测模型 - 训练与验证脚本
================================================================
数据集: Pima Indians Diabetes Database
        原始来源 National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
        经 UCI / OpenML (dataset id 37) 分发，逐字节校验见 data/DATA_PROVENANCE.md
样本:   768 条（全部为 ≥21 岁 Pima 印第安女性）
特征:   8 个（简洁的常规临床指标）
模型:   XGBoost 二分类
================================================================
用法:
    python scripts/train_model.py                    # 训练 + 验证 + 发布推理模型
    python scripts/train_model.py --data other.csv   # 使用外部 CSV
    python scripts/train_model.py --quick            # 跳过重复交叉验证（更快）
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SKILL_DIR, "data")
MODELS_DIR = os.path.join(SKILL_DIR, "models")
RESULTS_DIR = os.path.join(SKILL_DIR, "results")
DEFAULT_DATA = os.path.join(DATA_DIR, "pima_indians_diabetes.csv")

# ---------------------------------------------------------------------------
# 特征定义
# ---------------------------------------------------------------------------
FEATURES = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
TARGET = "label"

FEATURE_NAMES_CN = {
    "preg": "怀孕次数",
    "plas": "口服糖耐量试验2小时血糖浓度",
    "pres": "舒张压",
    "skin": "三头肌皮褶厚度",
    "insu": "2小时血清胰岛素",
    "mass": "BMI体重指数",
    "pedi": "糖尿病家族遗传函数",
    "age": "年龄",
}
FEATURE_UNITS = {
    "preg": "次", "plas": "mg/dL", "pres": "mm Hg", "skin": "mm",
    "insu": "mu U/ml", "mass": "kg/m^2", "pedi": "-", "age": "岁",
}

# 生理学上不可能为 0 的列：数据集用 0 表示缺失值
ZERO_AS_MISSING = ["plas", "pres", "skin", "insu", "mass"]

# ---------------------------------------------------------------------------
# 训练配置（全部固定，保证可复现）
# ---------------------------------------------------------------------------
SPLIT_CONFIG = {"test_size": 0.2, "random_state": 42, "stratify": True}
CV_CONFIG = {"n_splits": 5, "n_repeats": 10, "random_state": 42}
DECISION_THRESHOLD = 0.5

# 调参得出的 XGBoost 超参数（由训练集内层 5 折 GridSearchCV 按 AUC 选出）
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

# 验收标准：测试精度需达到的下限
ACCEPTANCE = {
    "holdout_accuracy": 0.75,
    "holdout_auc": 0.80,
    "cv_accuracy": 0.75,
    "cv_auc": 0.80,
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_dataframe(data_path=None):
    """加载数据，并把 5 个不可能为 0 的列中的 0 转为 NaN（视为缺失）。"""
    path = data_path or DEFAULT_DATA
    if not os.path.exists(path):
        raise SystemExit(
            f"错误: 未找到数据文件: {path}\n"
            f"请运行 python {os.path.join(SCRIPT_DIR, 'fetch_data.py')} 重新下载，"
            f"或使用 --data 指定 CSV。"
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "pregnancies": "preg", "glucose": "plas", "bloodpressure": "pres",
        "blood_pressure": "pres", "skinthickness": "skin", "skin_thickness": "skin",
        "insulin": "insu", "bmi": "mass",
        "diabetespedigreefunction": "pedi", "diabetes_pedigree_function": "pedi",
        "outcome": "label", "class": "label", "target": "label",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"错误: 数据缺少列: {missing}\n必需列: {FEATURES + [TARGET]}")

    df = df[FEATURES + [TARGET]].copy()

    # 记录缺失情况后，把 0 转为 NaN
    n_missing = {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING}
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info = {"path": os.path.abspath(path), "sha256": sha,
            "n_samples": int(len(df)), "zero_as_missing_counts": n_missing}
    return df, info


def build_model():
    """构造 中位数插补 + XGBoost 两个组件。"""
    return SimpleImputer(strategy="median"), xgb.XGBClassifier(**XGB_PARAMS)


def fit_pipeline(X_train, y_train):
    """在训练集上拟合插补器与模型，返回 (imputer, model)。"""
    imputer, model = build_model()
    Xt = imputer.fit_transform(X_train)
    model.fit(Xt, y_train)
    return imputer, model


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate_holdout(df, verbose=True):
    """80/20 分层留出测试集评估。"""
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=SPLIT_CONFIG["test_size"],
        random_state=SPLIT_CONFIG["random_state"],
        stratify=y if SPLIT_CONFIG["stratify"] else None,
    )
    imputer, model = fit_pipeline(X_tr, y_tr)

    proba = model.predict_proba(imputer.transform(X_te))[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_te, pred)

    metrics = {
        "auc": round(float(roc_auc_score(y_te, proba)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "threshold": DECISION_THRESHOLD,
    }

    if verbose:
        print(f"\n----- 留出测试集评估（{len(X_tr)} 训练 / {len(X_te)} 测试）-----")
        print(f"AUC:        {metrics['auc']:.4f}")
        print(f"准确率:      {metrics['accuracy']:.4f}")
        print(f"精确率:      {metrics['precision']:.4f}")
        print(f"召回率:      {metrics['recall']:.4f}")
        print(f"F1 分数:     {metrics['f1']:.4f}")
        print(f"混淆矩阵: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return metrics


def evaluate_cv(df, quick=False, verbose=True):
    """重复分层交叉验证（比单次留出更稳定），在全量数据上进行。"""
    from sklearn.pipeline import Pipeline

    X, y = df[FEATURES], df[TARGET]
    n_repeats = 2 if quick else CV_CONFIG["n_repeats"]
    cv = RepeatedStratifiedKFold(
        n_splits=CV_CONFIG["n_splits"], n_repeats=n_repeats,
        random_state=CV_CONFIG["random_state"],
    )

    imputer, model = build_model()
    pipe = Pipeline([("imputer", imputer), ("model", model)])
    scores = cross_validate(
        pipe, X, y, cv=cv, n_jobs=-1,
        scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
    )

    metrics = {}
    for k in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        v = scores[f"test_{k}"]
        metrics[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}
    metrics["n_splits"] = CV_CONFIG["n_splits"]
    metrics["n_repeats"] = n_repeats
    metrics["n_folds_total"] = CV_CONFIG["n_splits"] * n_repeats

    if verbose:
        print(f"\n----- {n_repeats}×{CV_CONFIG['n_splits']} 折重复交叉验证（全量 {len(X)} 样本）-----")
        for k in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            print(f"{k:10s}: {metrics[k]['mean']:.4f}  ± {metrics[k]['std']:.4f}")
    return metrics


def check_acceptance(holdout, cv, verbose=True):
    """检查是否达到验收标准。"""
    checks = {
        "holdout_accuracy": holdout["accuracy"] >= ACCEPTANCE["holdout_accuracy"],
        "holdout_auc": holdout["auc"] >= ACCEPTANCE["holdout_auc"],
        "cv_accuracy": cv["accuracy"]["mean"] >= ACCEPTANCE["cv_accuracy"],
        "cv_auc": cv["roc_auc"]["mean"] >= ACCEPTANCE["cv_auc"],
    }
    if verbose:
        labels = {
            "holdout_accuracy": f"留出测试集 准确率 >= {ACCEPTANCE['holdout_accuracy']:.2f}",
            "holdout_auc": f"留出测试集 AUC    >= {ACCEPTANCE['holdout_auc']:.2f}",
            "cv_accuracy": f"交叉验证   准确率 >= {ACCEPTANCE['cv_accuracy']:.2f}",
            "cv_auc": f"交叉验证   AUC    >= {ACCEPTANCE['cv_auc']:.2f}",
        }
        print("\n----- 验收标准检查 -----")
        for key, ok in checks.items():
            print(f"  [{'通过' if ok else '未通过'}] {labels[key]}")
        print(f"\n  综合结论: {'全部达标' if all(checks.values()) else '未达标'}")
    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def train_model(data_path=None, quick=False, verbose=True):
    df, data_info = load_dataframe(data_path)

    if verbose:
        print("=" * 68)
        print(" 糖尿病风险预测模型 - 训练与验证")
        print("=" * 68)
        print(f"数据文件: {data_info['path']}")
        print(f"SHA256  : {data_info['sha256']}")
        print(f"样本数  : {data_info['n_samples']}   特征数: {len(FEATURES)}")
        n_pos = int(df[TARGET].sum())
        print(f"类别分布: 阳性(糖尿病) {n_pos} / 阴性 {len(df) - n_pos}  "
              f"-> 阳性率 {n_pos / len(df):.1%}")
        print("\n0 视为缺失的列（生理上不可能为 0）:")
        for c, n in data_info["zero_as_missing_counts"].items():
            print(f"  {c:5s}: {n:3d} 个 ({n / len(df):.1%})")

    # 1) 留出测试集
    holdout = evaluate_holdout(df, verbose=verbose)

    # 2) 重复交叉验证
    cv = evaluate_cv(df, quick=quick, verbose=verbose)

    # 3) 验收
    acceptance = check_acceptance(holdout, cv, verbose=verbose)

    # 4) 用全量数据重训发布模型
    imputer, model = fit_pipeline(df[FEATURES], df[TARGET])
    n_pos = int(df[TARGET].sum())
    if verbose:
        print(f"\n----- 发布模型（用全量 {len(df)} 条样本重训）-----")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_json = os.path.join(MODELS_DIR, "diabetes_xgb_model.json")
    model_ubj = os.path.join(MODELS_DIR, "diabetes_xgb_model.ubj")
    model.save_model(model_json)
    model.save_model(model_ubj)

    try:
        import joblib

        joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))
    except ImportError:
        pass

    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["feature_cn"] = importance["feature"].map(FEATURE_NAMES_CN)
    importance["importance"] = importance["importance"].round(4)

    if verbose:
        print("\n----- 影响糖尿病的关键特征（按重要性排序）-----")
        for _, r in importance.iterrows():
            print(f"  {r['feature']:5s} ({r['feature_cn']}): {r['importance']:.4f}")

    medians = {c: round(float(m), 4) for c, m in zip(FEATURES, imputer.statistics_)}

    metadata = {
        "skill_name": "diabetes_prediction",
        "version": "2.0",
        "model_type": "xgboost.XGBClassifier",
        "task": "binary_classification",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "full_dataset",
        "dataset": {
            "name": "Pima Indians Diabetes Database",
            "original_owner": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
            "distribution": "UCI Machine Learning Repository / OpenML dataset id 37",
            "file": "data/pima_indians_diabetes.csv",
            "sha256": data_info["sha256"],
            "n_samples": data_info["n_samples"],
            "n_features": len(FEATURES),
            "features": FEATURES,
            "features_cn": FEATURE_NAMES_CN,
            "feature_units": FEATURE_UNITS,
            "target": TARGET,
            "target_labels": {"0": "无糖尿病", "1": "糖尿病"},
            "positive_count": n_pos,
            "population": ">=21 岁 Pima 印第安女性（数据集原始限制）",
        },
        "preprocessing": {
            "zero_as_missing_columns": ZERO_AS_MISSING,
            "imputer": "sklearn.impute.SimpleImputer(strategy='median')",
            "imputer_statistics": medians,
            "feature_order": FEATURES,
            "scaling": "无（XGBoost 对特征尺度不敏感）",
        },
        "split": {**SPLIT_CONFIG, "n_train": holdout["n_train"], "n_test": holdout["n_test"]},
        "cross_validation": CV_CONFIG,
        "hyperparameters": XGB_PARAMS,
        "decision_threshold": DECISION_THRESHOLD,
        "validation": {
            "holdout": holdout,
            "cross_validation": cv,
            "acceptance_criteria": ACCEPTANCE,
            "acceptance_passed": acceptance,
        },
        "artifacts": {
            "model_json": "diabetes_xgb_model.json",
            "model_ubj": "diabetes_xgb_model.ubj",
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
        print(f"  插补中位数  : {medians}")
        print("\n训练完成!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="糖尿病风险预测模型 - 训练与验证")
    parser.add_argument("--data", default=None, help="外部训练 CSV（默认使用 data/ 内置数据）")
    parser.add_argument("--quick", action="store_true", help="减少交叉验证重复次数，加快运行")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    meta = train_model(data_path=args.data, quick=args.quick, verbose=not args.quiet)
    if not all(meta["validation"]["acceptance_passed"].values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
