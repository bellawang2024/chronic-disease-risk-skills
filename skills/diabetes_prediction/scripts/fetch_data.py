#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病数据集下载与校验脚本
================================================================
从 OpenML 官方接口下载 Pima Indians Diabetes Database，校验 MD5 后
转换为 data/pima_indians_diabetes.csv。

数据溯源:
  原始所有者: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
  分发渠道  : OpenML dataset id 37 (https://www.openml.org/d/37)
              原始 UCI 页面: https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes

用法:
    python scripts/fetch_data.py            # 下载并校验
    python scripts/fetch_data.py --check    # 仅校验已有文件，不下载
================================================================
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SKILL_DIR, "data")
OUT_CSV = os.path.join(DATA_DIR, "pima_indians_diabetes.csv")

OPENML_ARFF_URL = "https://openml.org/data/v1/download/37/diabetes.arff"

# OpenML 官方公布的校验值（用 md5 校验原始 ARFF）
EXPECTED_MD5 = "3cbaa3e54586aa88cf6aacb4033e4470"
EXPECTED_N_SAMPLES = 768

HEADER = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "label"]


def arff_to_csv(text):
    """把 OpenML 的 ARFF 文本转换为干净 CSV 内容。"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("@"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue
        label = 1 if parts[-1] == "tested_positive" else 0
        rows.append(parts[:-1] + [str(label)])
    return rows


def download():
    print(f"下载: {OPENML_ARFF_URL}")
    req = urllib.request.Request(OPENML_ARFF_URL, headers={"User-Agent": "diabetes-skill/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()

    md5 = hashlib.md5(raw).hexdigest()
    print(f"MD5    : {md5}")
    print(f"期望   : {EXPECTED_MD5}")
    if md5 != EXPECTED_MD5:
        print("错误: MD5 校验失败，数据可能被篡改或已更新，已中止。", file=sys.stderr)
        sys.exit(1)
    print("MD5 校验通过")

    rows = arff_to_csv(raw.decode("utf-8"))
    if len(rows) != EXPECTED_N_SAMPLES:
        print(f"错误: 样本数 {len(rows)} != 期望 {EXPECTED_N_SAMPLES}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    sha = hashlib.sha256(open(OUT_CSV, "rb").read()).hexdigest()
    print(f"已写出: {OUT_CSV}")
    print(f"样本数: {len(rows)}")
    print(f"CSV SHA256: {sha}")


def check():
    if not os.path.exists(OUT_CSV):
        print(f"错误: 文件不存在: {OUT_CSV}", file=sys.stderr)
        sys.exit(1)
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    print(f"文件      : {OUT_CSV}")
    print(f"表头      : {header}")
    print(f"样本数    : {len(data)}")
    print(f"SHA256    : {hashlib.sha256(open(OUT_CSV, 'rb').read()).hexdigest()}")
    ok = header == HEADER and len(data) == EXPECTED_N_SAMPLES
    print(f"校验结果  : {'通过' if ok else '不通过'}")
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载并校验 Pima 糖尿病数据集")
    parser.add_argument("--check", action="store_true", help="仅校验已有文件，不下载")
    args = parser.parse_args()
    check() if args.check else download()


if __name__ == "__main__":
    main()
