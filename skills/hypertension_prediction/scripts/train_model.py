#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高血压风险预测模型 - 训练与验证脚本
================================================================
数据来源: NHANES 2017-2018 (CDC / NCHS)
          美国非机构化平民人口全国代表性抽样 —— 全人群, 不限性别/种族
样本:     5,250 名成人 (>=18 岁, 有有效血压测量)
特征:     16 个常规体检 / 问卷 / 血检指标
标签:     高血压 (ACC/AHA 2017: 平均 SBP>=130 或 DBP>=80 或 正在服降压药)
模型:     XGBoost 二分类
================================================================
用法:
    python scripts/train_model.py
    python scripts/train_model.py --quick
    python scripts/train_model.py --include-sensitive # 加入性别/种族特征（默认不含）
    python scripts/train_model.py --data other.csv
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
DEFAULT_DATA = os.path.join(DATA_DIR, "hypertension_nhanes.csv")

# ---------------------------------------------------------------------------
# 特征定义
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "age", "sex_male", "bmi", "waist_cm", "height_cm", "race", "education",
    "income_pir", "smoker", "diabetes", "pulse", "hba1c", "cholesterol",
    "hdl", "creatinine", "uric_acid",
]
# 敏感/人群划分属性（默认移除，可用 --include-sensitive 加入）
SENSITIVE_FEATURES = ["sex_male", "race"]
TARGET = "label"

FEATURE_NAMES_CN = {
    "age": "年龄",
    "sex_male": "性别(1=男,0=女)",
    "bmi": "BMI体重指数",
    "waist_cm": "腰围",
    "height_cm": "身高",
    "race": "种族/族裔(1=墨西哥裔 2=其他西语裔 3=非西语裔白人 4=非西语裔黑人 6=非西语裔亚裔)",
    "education": "教育程度(1=低于高中 2=高中 3=大专 4=本科 5=研究生)",
    "income_pir": "家庭收入贫困比",
    "smoker": "吸烟(一生≥100支)",
    "diabetes": "糖尿病(自报,含临界)",
    "pulse": "静息脉搏",
    "hba1c": "糖化血红蛋白HbA1c",
    "cholesterol": "总胆固醇",
    "hdl": "高密度脂蛋白HDL",
    "creatinine": "血清肌酐",
    "uric_acid": "血清尿酸",
}
FEATURE_UNITS = {
    "age": "岁", "sex_male": "1/0", "bmi": "kg/m^2", "waist_cm": "cm",
    "height_cm": "cm", "race": "编码", "education": "编码", "income_pir": "比值",
    "smoker": "1/0", "diabetes": "1/0", "pulse": "次/分", "hba1c": "%",
    "cholesterol": "mg/dL", "hdl": "mg/dL", "creatinine": "mg/dL", "uric_acid": "mg/dL",
}
FEATURE_RANGES = {
    "age": (18, 80), "sex_male": (0, 1), "bmi": (10.0, 90.0), "waist_cm": (40.0, 200.0),
    "height_cm": (120.0, 220.0), "race": (1, 6), "education": (1, 5),
    "income_pir": (0.0, 5.0), "smoker": (0, 1), "diabetes": (0, 1),
    "pulse": (20, 200), "hba1c": (3.0, 20.0), "cholesterol": (50.0, 600.0),
    "hdl": (5.0, 200.0), "creatinine": (0.1, 20.0), "uric_acid": (0.5, 20.0),
}

# ---------------------------------------------------------------------------
# 训练配置（固定，保证可复现）
# ---------------------------------------------------------------------------
SPLIT_CONFIG = {"test_size": 0.2, "random_state": 42, "stratify": True}
CV_CONFIG = {"n_splits": 5, "n_repeats": 10, "random_state": 42}
DECISION_THRESHOLD = 0.5

# 由训练集内层 5 折 GridSearchCV 按 AUC 选出的超参数
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

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
    info = {"path": os.path.abspath(path), "sha256": sha, "n_samples": int(len(df))}
    return df, info


def feature_list(include_sensitive=False):
    """默认移除性别/种族等敏感属性，使模型不依赖人群划分属性。

    实测（NHANES 2017-2018, 10×5 折重复交叉验证）：
      含性别/种族 16 特征 -> AUC 0.8301, 准确率 0.7594
      不含       14 特征 -> AUC 0.8234, 准确率 0.7597
    准确率基本持平、AUC 仅低 0.007，因此默认采用不含敏感属性的版本。
    """
    if include_sensitive:
        return list(ALL_FEATURES)
    return [f for f in ALL_FEATURES if f not in SENSITIVE_FEATURES]


def fit_pipeline(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X_train)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xt, y_train)
    return imputer, model


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


def evaluate_cv(df, features, quick=False, verbose=True):
    from sklearn.pipeline import Pipeline

    X, y = df[features], df[TARGET]
    n_repeats = 2 if quick else CV_CONFIG["n_repeats"]
    cv = RepeatedStratifiedKFold(
        n_splits=CV_CONFIG["n_splits"], n_repeats=n_repeats,
        random_state=CV_CONFIG["random_state"],
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", xgb.XGBClassifier(**XGB_PARAMS))])
    scores = cross_validate(
        pipe, X, y, cv=cv, n_jobs=-1,
        scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
    )
    metrics = {k: {"mean": round(float(scores[f"test_{k}"].mean()), 4),
                   "std": round(float(scores[f"test_{k}"].std()), 4)}
               for k in ["roc_auc", "accuracy", "precision", "recall", "f1"]}
    metrics.update({"n_splits": CV_CONFIG["n_splits"], "n_repeats": n_repeats,
                    "n_folds_total": CV_CONFIG["n_splits"] * n_repeats})

    if verbose:
        print(f"\n----- {n_repeats}×{CV_CONFIG['n_splits']} 折重复交叉验证（全量 {len(X)} 样本）-----")
        for k in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            print(f"{k:10s}: {metrics[k]['mean']:.4f}  ± {metrics[k]['std']:.4f}")
    return metrics


def check_acceptance(holdout, cv, verbose=True):
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
def train_model(data_path=None, quick=False, include_sensitive=False, verbose=True):
    df, data_info = load_dataframe(data_path)
    features = feature_list(include_sensitive)

    if verbose:
        n_pos = int(df[TARGET].sum())
        print("=" * 70)
        print(" 高血压风险预测模型 - 训练与验证")
        print("=" * 70)
        print(f"数据文件: {data_info['path']}")
        print(f"SHA256  : {data_info['sha256']}")
        print(f"样本数  : {data_info['n_samples']}   特征数: {len(features)}")
        print(f"类别分布: 阳性(高血压) {n_pos} / 阴性 {len(df) - n_pos}  "
              f"-> 患病率 {n_pos / len(df):.1%}")
        print(f"人群构成: 男 {int(df.sex_male.sum())} / 女 {int((1 - df.sex_male).sum())} "
              f"(全人群抽样，未按性别划分)")
        print(f"敏感属性特征: {'已包含 ' + str(SENSITIVE_FEATURES) if include_sensitive else '已移除 ' + str(SENSITIVE_FEATURES)}")

    holdout = evaluate_holdout(df, features, verbose=verbose)
    cv = evaluate_cv(df, features, quick=quick, verbose=verbose)
    acceptance = check_acceptance(holdout, cv, verbose=verbose)

    imputer, model = fit_pipeline(df[features], df[TARGET])
    if verbose:
        print(f"\n----- 发布模型（用全量 {len(df)} 条样本重训）-----")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_json = os.path.join(MODELS_DIR, "hypertension_xgb_model.json")
    model_ubj = os.path.join(MODELS_DIR, "hypertension_xgb_model.ubj")
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
        print("\n----- 影响高血压的关键特征（按重要性排序）-----")
        for _, r in importance.iterrows():
            print(f"  {r['feature']:12s} ({r['feature_cn'][:20]}): {r['importance']:.4f}")

    medians = {f: round(float(v), 4) for f, v in zip(features, imputer.statistics_)}
    n_pos = int(df[TARGET].sum())

    metadata = {
        "skill_name": "hypertension_prediction",
        "version": "1.0",
        "model_type": "xgboost.XGBClassifier",
        "task": "binary_classification",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "full_dataset",
        "dataset": {
            "name": "NHANES 2017-2018 (National Health and Nutrition Examination Survey)",
            "owner": "U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)",
            "survey_cycle": "2017-2018 (cycle J)",
            "target_population": "美国非机构化平民人口（全人群抽样，不限性别、种族、地区）",
            "cohort_definition": "成人(>=18 岁)且有有效血压测量",
            "file": "data/hypertension_nhanes.csv",
            "sha256": data_info["sha256"],
            "n_samples": data_info["n_samples"],
            "n_features": len(features),
            "features": features,
            "features_cn": FEATURE_NAMES_CN,
            "feature_units": FEATURE_UNITS,
            "sensitive_features_included": include_sensitive,
            "target": TARGET,
            "target_definition": "ACC/AHA 2017: 平均 SBP>=130 或 平均 DBP>=80 或 正在服用降压药 (BPQ050A=1)",
            "target_labels": {"0": "无高血压", "1": "高血压"},
            "positive_count": n_pos,
            "prevalence": round(n_pos / len(df), 4),
            "n_male": int(df.sex_male.sum()),
            "n_female": int((1 - df.sex_male).sum()),
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
        "decision_threshold": DECISION_THRESHOLD,
        "validation": {
            "holdout": holdout,
            "cross_validation": cv,
            "acceptance_criteria": ACCEPTANCE,
            "acceptance_passed": acceptance,
        },
        "artifacts": {
            "model_json": "hypertension_xgb_model.json",
            "model_ubj": "hypertension_xgb_model.ubj",
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
        print("\n训练完成!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="高血压风险预测模型 - 训练与验证")
    parser.add_argument("--data", default=None, help="外部训练 CSV")
    parser.add_argument("--quick", action="store_true", help="减少交叉验证重复次数")
    parser.add_argument("--include-sensitive", action="store_true",
                        help=f"把敏感属性特征 {SENSITIVE_FEATURES} 加入模型（默认移除）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    meta = train_model(data_path=args.data, quick=args.quick,
                       include_sensitive=args.include_sensitive, verbose=not args.quiet)
    if not all(meta["validation"]["acceptance_passed"].values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
