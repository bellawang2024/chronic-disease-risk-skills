#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑卒中数据集下载、校验与构建脚本
================================================================
从美国 CDC 官方站点下载 BRFSS 2017 原始数据，校验 SHA256 后构建
脑卒中建模队列 data/stroke_brfss.csv。

数据溯源:
  调查项目: BRFSS (Behavioral Risk Factor Surveillance System)
  执行机构: U.S. Centers for Disease Control and Prevention (CDC)
  调查年份: 2017
  目标人群: 美国 50 州及属地的成年居民（全人群电话调查，不限性别/种族）
  官方页面: https://www.cdc.gov/brfss/annual_data/annual_2017.html

注意:
  1. 压缩包解压后的文件名含**尾随空格**（LLCP2017.XPT ），脚本会自动处理。
  2. CDC 的 CDN 会拒绝浏览器/自定义 User-Agent（返回 403），仅 curl 风格可通过，
     脚本内置多 UA 回退与 curl 兜底。
  3. 原始 XPT 解压后约 1.29 GB，解析约需 50 秒与约 2 GB 内存。

用法:
    python scripts/fetch_data.py            # 下载 + 校验 + 构建
    python scripts/fetch_data.py --check    # 仅校验已构建的 CSV
    python scripts/fetch_data.py --keep-raw # 保留下载的原始文件
================================================================
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SKILL_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_CSV = os.path.join(DATA_DIR, "stroke_brfss.csv")

BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip"
EXPECTED_ZIP_SHA256 = "ed692aae6a5b240c07fe186f0397b874553b805dcc83ab043c234386003bea07"
EXPECTED_CSV_SHA256 = "360136a4ea1b6888315c671975e38b6ad5053f2afdb74cbf824f26239158b238"
EXPECTED_N_SAMPLES = 448666
EXPECTED_N_RAW = 450016

OUTPUT_COLUMNS = [
    "age", "bmi", "height_cm", "weight_kg", "smoker", "diabetes",
    "hypertension", "high_chol", "gen_health", "exercise", "education",
    "income", "drinks_wk", "depression", "copd", "label",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yes_no(v, yes_value):
    """把 1/2 编码映射为 1/0，显式指定代表 Yes 的取值，避免方向搞反。"""
    no_value = 1.0 if yes_value == 2.0 else 2.0
    return v.map({yes_value: 1.0, no_value: 0.0})


def _download_to(url, dest):
    """下载到 dest。

    CDC 的 CDN 对 User-Agent 有过滤：实测浏览器 UA 与自定义 UA 均返回 403，
    仅 curl 风格 UA 可通过。因此依次尝试多种 UA，全部失败时回退到 curl 命令。
    """
    agents = ["curl/8.7.1", "curl/7.88.1", "Wget/1.21.3", "python-requests/2.31.0"]
    last_err = None
    for ua in agents:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return "urllib(UA=%s)" % ua
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    import shutil
    import subprocess

    if shutil.which("curl"):
        r = subprocess.run(["curl", "-sSL", "--retry", "2", "-o", dest, url],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return "curl"
        last_err = RuntimeError(r.stderr.strip() or f"curl 退出码 {r.returncode}")

    raise RuntimeError(f"下载失败: {last_err}")


def download_raw():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "LLCP2017XPT.zip")

    need = True
    if os.path.exists(zip_path) and _sha256(zip_path) == EXPECTED_ZIP_SHA256:
        print("  压缩包已存在且校验通过，跳过下载")
        need = False
    if need:
        print(f"  下载 {BRFSS_URL}")
        print("  （约 107 MB，视网络情况可能需要 1-2 分钟）")
        how = _download_to(BRFSS_URL, zip_path)
        print(f"  下载完成（方式: {how}）")
        actual = _sha256(zip_path)
        if actual != EXPECTED_ZIP_SHA256:
            print(f"错误: 压缩包 SHA256 校验失败\n  期望 {EXPECTED_ZIP_SHA256}\n  实际 {actual}",
                  file=sys.stderr)
            sys.exit(1)
        print("  SHA256 校验通过")

    print("  解压中（原始 XPT 约 1.29 GB）...")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        # 注意: BRFSS 压缩包内文件名为 'LLCP2017.XPT '，带一个尾随空格，
        # 因此必须先 strip 再判断后缀。
        def _is_xpt(n):
            base = os.path.basename(n).strip().upper()
            return base.startswith("LLCP2017") and base.endswith(".XPT")

        target = next((n for n in names if _is_xpt(n)), None)
        if target is None:
            print(f"错误: 压缩包内未找到 XPT 文件，内容: {names}", file=sys.stderr)
            sys.exit(1)
        xpt_path = os.path.join(RAW_DIR, "LLCP2017.XPT")
        with z.open(target) as src, open(xpt_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    return xpt_path


def build_cohort(xpt_path):
    print("  解析 XPT（约需 50 秒）...")
    d = pd.read_sas(xpt_path, format="xport")
    if len(d) != EXPECTED_N_RAW:
        print(f"警告: 原始记录数 {len(d):,} 与预期的 {EXPECTED_N_RAW:,} 不一致", file=sys.stderr)

    out = pd.DataFrame({
        "age":          d._AGE80,
        "bmi":          d._BMI5 / 100,
        "height_cm":    d.HTM4,
        "weight_kg":    d.WTKG3 / 100,
        "smoker":       d._SMOKER3.where(d._SMOKER3.between(1, 4)),
        "diabetes":     d.DIABETE3.map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0}),
        "hypertension": yes_no(d._RFHYPE5, 2.0),
        "high_chol":    yes_no(d._RFCHOL1, 2.0),
        "gen_health":   d.GENHLTH.where(d.GENHLTH.between(1, 5)),
        "exercise":     yes_no(d._TOTINDA, 1.0),
        "education":    d.EDUCA.where(d.EDUCA.between(1, 6)),
        "income":       d.INCOME2.where(d.INCOME2.between(1, 8)),
        # _DRNKWEK 为 杯/周×100；非饮酒者编码为极小非规格化浮点数，按 0 处理
        "drinks_wk":    (d._DRNKWEK / 100).where(d._DRNKWEK < 99900)
                        .mask(lambda x: x < 0.005, 0.0),
        "depression":   yes_no(d.ADDEPEV2, 1.0),
        "copd":         yes_no(d.CHCCOPD1, 1.0),
        "label":        d.CVDSTRK3.map({1.0: 1.0, 2.0: 0.0}),
    })
    return out.dropna(subset=["label"])[OUTPUT_COLUMNS]


def download(keep_raw=False):
    print("获取 BRFSS 2017 原始数据 ...")
    xpt_path = download_raw()

    print("\n构建建模队列 ...")
    out = build_cohort(xpt_path)
    if len(out) != EXPECTED_N_SAMPLES:
        print(f"警告: 队列样本数 {len(out):,} 与构建时的 {EXPECTED_N_SAMPLES:,} 不一致",
              file=sys.stderr)

    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")
    sha = _sha256(OUT_CSV)

    print(f"\n已写出: {OUT_CSV}")
    print(f"形状  : {out.shape[0]:,} 行 × {out.shape[1]} 列")
    print(f"患病率: {out.label.mean():.2%}")
    print(f"SHA256: {sha}")
    print("与构建时逐字节一致" if sha == EXPECTED_CSV_SHA256
          else "注意: 与构建时校验值不同（可能因 pandas 版本差异），请以实际训练结果为准")

    if not keep_raw:
        import shutil

        shutil.rmtree(RAW_DIR, ignore_errors=True)


def check():
    if not os.path.exists(OUT_CSV):
        print(f"错误: 文件不存在: {OUT_CSV}\n请先运行 python scripts/fetch_data.py",
              file=sys.stderr)
        sys.exit(1)
    with open(OUT_CSV, encoding="utf-8") as f:
        header = [c.strip() for c in f.readline().split(",")]
        n = sum(1 for _ in f)
    print(f"文件    : {OUT_CSV}")
    print(f"列      : {header}")
    print(f"样本数  : {n:,}")
    print(f"SHA256  : {_sha256(OUT_CSV)}")
    ok = header == OUTPUT_COLUMNS and n == EXPECTED_N_SAMPLES
    print(f"校验结果: {'通过' if ok else '不通过'}")
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载并构建 BRFSS 脑卒中数据集")
    parser.add_argument("--check", action="store_true", help="仅校验已构建的 CSV")
    parser.add_argument("--keep-raw", action="store_true", help="保留下载的原始文件")
    args = parser.parse_args()
    check() if args.check else download(keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
