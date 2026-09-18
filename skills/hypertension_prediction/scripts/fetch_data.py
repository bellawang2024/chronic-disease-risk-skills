#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高血压数据集下载、校验与构建脚本
================================================================
从美国 CDC / NCHS 官方站点下载 NHANES 2017-2018 原始数据文件，
逐文件校验 SHA256 后构建高血压预测建模队列
data/hypertension_nhanes.csv。

数据溯源:
  调查项目: NHANES (National Health and Nutrition Examination Survey)
  执行机构: U.S. Centers for Disease Control and Prevention (CDC),
            National Center for Health Statistics (NCHS)
  调查周期: 2017-2018 (cycle J)
  目标人群: 美国非机构化平民人口（全人群，不限性别/种族/年龄）
  官方页面: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

用法:
    python scripts/fetch_data.py            # 下载 + 校验 + 构建
    python scripts/fetch_data.py --check    # 仅校验已构建的 CSV
    python scripts/fetch_data.py --keep-raw # 保留原始 xpt 文件到 data/raw/
================================================================
"""
import argparse
import csv
import hashlib
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SKILL_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_CSV = os.path.join(DATA_DIR, "hypertension_nhanes.csv")
TMP_DIR = os.path.join(RAW_DIR, ".tmp")

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt"

# 文件名 -> 官方公布的 SHA256（构建时逐一下载校验所得，可用于检测文件变更）
RAW_FILES = {
    "DEMO_J":   "c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec",
    "BPX_J":    "f98f1d9a4c18173e715c749b4294f6f1525140d5ad5751316bb17cc038fb3423",
    "BPQ_J":    "63cfe1c331a1e7d3534328ac86312a21ffee3e01aa189bc1f74d06855239e5aa",
    "BMX_J":    "8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838a",
    "SMQ_J":    "73c3708e540cbf520a566508aa629c3e4ba53cad77dae9388ca36384646455f4",
    "DIQ_J":    "1ecbf5360dfc331d1efbf32198553dc30e9a1f4cff0a907ed30ca72bac797f89",
    "GHB_J":    "35f07094573a0061a03ed609a5a363b34eb1b1c7065d1623b43d72e132a8a654",
    "TCHOL_J":  "0291a6a4f6d82dac8392c8c992b519946824dfeda01647935520cf506ed9f4ab",
    "HDL_J":    "9f4eb41b89f0c9f3d6262f873e671eeb2f2eb40581a8426b2a3c8fd4fc19439e",
    "BIOPRO_J": "5bcd5722c1892883b96a9d7fed0befadabacae313f800fadc0133cd5dd00c4c6",
}

EXPECTED_N_SAMPLES = 5250
EXPECTED_CSV_SHA256 = "e08ec328a3cd6cec11b8beee90352753cc3db887857944c14a591252e1b2be5f"

OUTPUT_COLUMNS = [
    "age", "sex_male", "bmi", "waist_cm", "height_cm", "race", "education",
    "income_pir", "smoker", "diabetes", "pulse", "hba1c", "cholesterol",
    "hdl", "creatinine", "uric_acid", "label",
]


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def download_raw():
    os.makedirs(TMP_DIR, exist_ok=True)
    paths = {}
    for name, expected in RAW_FILES.items():
        path = os.path.join(TMP_DIR, f"{name}.xpt")
        if os.path.exists(path) and _sha256(path) == expected:
            print(f"  {name}.xpt  已存在且校验通过，跳过下载")
            paths[name] = path
            continue

        url = BASE_URL.format(name=name)
        print(f"  下载 {name}.xpt ...")
        req = urllib.request.Request(url, headers={"User-Agent": "hypertension-skill/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)

        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            print(f"错误: {name}.xpt SHA256 校验失败\n  期望 {expected}\n  实际 {actual}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"    SHA256 校验通过 ({len(data):,} bytes)")
        paths[name] = path
    return paths


def build_cohort(paths):
    """按统一定义构建建模队列（与训练脚本中的说明保持一致）。"""
    frames = {name: pd.read_sas(path, format="xport") for name, path in paths.items()}
    d = frames["DEMO_J"]
    for name, fr in frames.items():
        if name != "DEMO_J":
            d = d.merge(fr, on="SEQN", how="left")

    # 血压：取第 2~4 次读数均值（NHANES 标准做法，降低单次测量噪声），
    # 缺失时回退到第 1 次读数
    d["SBP"] = d[["BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1, skipna=True).fillna(d["BPXSY1"])
    d["DBP"] = d[["BPXDI2", "BPXDI3", "BPXDI4"]].mean(axis=1, skipna=True).fillna(d["BPXDI1"])

    # 高血压判定（ACC/AHA 2017 标准）：平均 SBP>=130 或 平均 DBP>=80 或 正在服用降压药
    d["HTN"] = ((d.SBP >= 130) | (d.DBP >= 80) | (d.BPQ050A == 1)).astype(int)

    # 队列纳入：成人(>=18) 且有有效血压测量
    coh = d[(d.RIDAGEYR >= 18) & d.SBP.notna() & d.DBP.notna()].copy()

    out = pd.DataFrame({
        "age":         coh.RIDAGEYR,
        "sex_male":    (coh.RIAGENDR == 1).astype(int),
        "bmi":         coh.BMXBMI,
        "waist_cm":    coh.BMXWAIST,
        "height_cm":   coh.BMXHT,
        "race":        coh.RIDRETH3.where(coh.RIDRETH3.isin([1, 2, 3, 4, 6])),
        "education":   coh.DMDEDUC2.where(coh.DMDEDUC2.isin([1, 2, 3, 4, 5])),
        "income_pir":  coh.INDFMPIR,
        "smoker":      (coh.SMQ020 == 1).astype(int),
        "diabetes":    coh.DIQ010.map({1.0: 1, 2.0: 0, 3.0: 0}),
        "pulse":       coh.BPXPLS,
        "hba1c":       coh.LBXGH,
        "cholesterol": coh.LBXTC,
        "hdl":         coh.LBDHDD,
        "creatinine":  coh.LBXSCR,
        "uric_acid":   coh.LBXSUA,
        "label":       coh.HTN,
    })
    return out[OUTPUT_COLUMNS]


def download(keep_raw=False):
    print("下载 NHANES 2017-2018 原始数据文件 ...")
    paths = download_raw()

    print("\n构建建模队列 ...")
    out = build_cohort(paths)

    if len(out) != EXPECTED_N_SAMPLES:
        print(f"警告: 队列样本数 {len(out)} 与构建时的 {EXPECTED_N_SAMPLES} 不一致", file=sys.stderr)

    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")
    sha = _sha256(OUT_CSV)

    print(f"\n已写出: {OUT_CSV}")
    print(f"形状  : {out.shape[0]} 行 × {out.shape[1]} 列")
    print(f"患病率: {out.label.mean():.1%}")
    print(f"SHA256: {sha}")
    if sha == EXPECTED_CSV_SHA256:
        print("与构建时逐字节一致 ✓")
    else:
        print("注意: 与构建时校验值不同（可能因 pandas 版本差异），请以实际训练结果为准")

    if not keep_raw:
        import shutil

        shutil.rmtree(TMP_DIR, ignore_errors=True)


def check():
    if not os.path.exists(OUT_CSV):
        print(f"错误: 文件不存在: {OUT_CSV}\n请先运行 python scripts/fetch_data.py",
              file=sys.stderr)
        sys.exit(1)
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    print(f"文件    : {OUT_CSV}")
    print(f"列      : {header}")
    print(f"样本数  : {len(data)}")
    print(f"SHA256  : {_sha256(OUT_CSV)}")
    ok = header == OUTPUT_COLUMNS and len(data) == EXPECTED_N_SAMPLES
    print(f"校验结果: {'通过' if ok else '不通过'}")
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载并构建 NHANES 高血压数据集")
    parser.add_argument("--check", action="store_true", help="仅校验已构建的 CSV")
    parser.add_argument("--keep-raw", action="store_true", help="保留原始 xpt 文件")
    args = parser.parse_args()
    check() if args.check else download(keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
