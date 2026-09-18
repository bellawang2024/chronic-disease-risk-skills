#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高脂血症（高胆固醇）数据集下载、校验与构建脚本
================================================================
从美国 CDC/NCHS 官方 FTP 下载 NHIS 2016–2025 Sample Adult 公开微数据，
逐文件校验 SHA256 后构建高脂血症建模队列 data/hyperlipidemia_nhis.csv。

数据溯源:
  调查项目: NHIS (National Health Interview Survey)
  执行机构: U.S. CDC, National Center for Health Statistics (NCHS)
  调查年份: 2016–2025（共 10 个年度）
  目标人群: 美国非机构化平民人口（全人群住户抽样，**男女兼备、不限种族**）
  许可    : 美国政府公开数据（Public Domain），**无需注册、无需数据使用协议**

标签（高胆固醇）跨问卷时代对应:
  2016–2018: CHLEV
  2019–2025: CHLEV_A
  官方问题: "Have you EVER been told by a doctor or other health professional
             that you had high cholesterol?"
  官方定义域: HHSTAT_A=1 / ASTATFLG='1' and AGE GE 18  → **全部 18 岁以上成人**
  编码: 1=Yes(阳性)  2=No(阴性)  7/8/9=拒答/未查明/不知道(缺失)

⚠️ 标签口径的重要说明:
  NHIS 只询问「高**胆固醇**」，并**没有**询问甘油三酯。
  因此本模型的标签严格来说是「曾被医生告知高胆固醇」，
  是三酰甘油正常者也可能被判定为高脂血症的**胆固醇部分**，
  不覆盖「单纯高甘油三酯血症」。详见 data/DATA_PROVENANCE.md。

为什么只用风险因素、不含血脂检验值:
  1. NHIS 为问卷调查，本身不含血脂检验数据（天然满足该要求）。
  2. 更重要的是**逻辑**：总胆固醇 / LDL-C 的升高**本身就是高脂血症的诊断依据**，
     用它们预测高脂血症近乎循环论证。故特征只取风险因素。
  3. 同理，**降脂药物使用史**（CHLMED_A / CHLMDEV2 等）与
     「近 12 个月是否查过血脂」（CHL12M_A）也**不纳入** —— 服药本身即意味着已确诊。

用法:
    python scripts/fetch_data.py            # 下载 + 校验 + 构建（约 45 MB）
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
OUT_CSV = os.path.join(DATA_DIR, "hyperlipidemia_nhis.csv")

BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS"

# 每年度的官方文件名与 SHA256（下载后逐一校验，不符即中止）
SOURCES = {
    2016: ("samadultcsv.zip", "49c635bf5c3612e9ac255ac8203bc1019c2df49d06709beaf4ee783e9dc0fd23"),
    2017: ("samadultcsv.zip", "4475d0e828c12d24c0d98d550751d8eb26723fd40b1ab9321fc9c3ab0fbef057"),
    2018: ("samadultcsv.zip", "e654fda5bb891315bd9fb4ac21bdcff86f3c9b78bd55738608f8567822e5a9a0"),
    2019: ("Adult19csv.zip", "fc105b045c46d871b029c5e6cefa797450fff9e4610fdc9a29e2af3137f16917"),
    2020: ("adult20csv.zip", "a517a49fa93285bef61ff4e849a0d4b5aa6fa99a1998c6caa716bfc9b5d984fe"),
    2021: ("adult21csv.zip", "6e71f5a34115f9aef51304957c08b6db7a91f8a0a6eb3d9d21cd4f254cbdbf78"),
    2022: ("adult22csv.zip", "25083298173acfff35c6635be0fbcaa3ff26b985e0a48fc5e7d9788761864dba"),
    2023: ("adult23csv.zip", "e6f0918e683e1d756e4298470737b4c99777e8a90ea9ee7488544b917f058585"),
    2024: ("adult24csv.zip", "5f926b2ec0af508fa84ebc791c83bfefa6f020724754aab599004706981abdaf"),
    2025: ("adult25csv.zip", "94d5477ffc3a03e292a3000da8eb5ad71626ae8032bbaa98c9711b7a3b572570"),
}

EXPECTED_CSV_SHA256 = "6c4055d80df9acc6b297f5581cebc625a90d0838066b77fb35fe5559f4198de1"
EXPECTED_N_SAMPLES = 291295
EXPECTED_N_POSITIVE = 91932

OUTPUT_COLUMNS = ["age", "bmi", "height_cm", "weight_kg", "smoker",
                  "hypertension", "diabetes", "label"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _download(url, dest):
    for ua in ["curl/8.7.1", "Wget/1.21.3", "python-requests/2.31.0"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=1800) as r, open(dest, "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    f.write(c)
            return f"urllib(UA={ua})"
        except Exception:  # noqa: BLE001
            continue
    import shutil
    import subprocess

    if shutil.which("curl"):
        r = subprocess.run(["curl", "-sSL", "--retry", "3", "-o", dest, url],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return "curl"
    raise RuntimeError(f"下载失败: {url}")


def download_raw():
    os.makedirs(RAW_DIR, exist_ok=True)
    paths = {}
    for year, (fname, expected) in SOURCES.items():
        z = os.path.join(RAW_DIR, f"{year}.zip")
        if os.path.exists(z) and _sha256(z) == expected:
            print(f"  {year}: 已存在且校验通过")
        else:
            url = f"{BASE}/{year}/{fname}"
            print(f"  下载 {year}: {fname}")
            how = _download(url, z)
            actual = _sha256(z)
            if actual != expected:
                print(f"错误: {year} SHA256 校验失败\n  期望 {expected}\n  实际 {actual}",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  {year}: 校验通过（{how}）")
        paths[year] = z
    return paths


# ---------------------------------------------------------------------------
# 特征构建
# ---------------------------------------------------------------------------
def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.strip().replace("", np.nan),
                         errors="coerce")


def _get(d, *names):
    for n in names:
        if n in d.columns:
            return d[n]
    return pd.Series(np.nan, index=d.index)


def _yn(s):
    """1=是 → 1；2=否 → 0；7/8/9 → 缺失"""
    return _num(s).map({1.0: 1.0, 2.0: 0.0})


def harmonize(d, year):
    """按问卷时代（<2019 / >=2019）统一为建模列。"""
    modern = year >= 2019

    def height(s):
        # NHIS 特殊码：96/97/98/99 英寸表示身高不可得
        return _num(s).where(lambda x: x.between(36, 90))

    def weight(s):
        # NHIS 特殊码：996-998 磅表示体重不可得
        return _num(s).where(lambda x: x.between(60, 900))

    if modern:
        h, w = height(_get(d, "HEIGHTTC_A")), weight(_get(d, "WEIGHTLBTC_A"))
        ev, now = _num(_get(d, "SMKEV_A")), _num(_get(d, "SMKNOW_A"))
        smk = pd.Series(np.nan, index=d.index)
        smk[ev == 2] = 4.0                                  # 从不
        smk[(ev == 1) & (now == 1)] = 1.0                   # 每天
        smk[(ev == 1) & (now == 2)] = 2.0                   # 偶尔
        smk[(ev == 1) & (now == 3)] = 3.0                   # 已戒
        chol = _num(_get(d, "CHLEV_A"))
    else:
        h, w = height(_get(d, "AHEIGHT")), weight(_get(d, "AWEIGHTP"))
        # SMKSTAT2 是 2016-2018 的派生 4 级变量，编码与本模型完全一致
        st2 = _num(_get(d, "SMKSTAT2"))
        smk = st2.where(st2.between(1, 4))
        chol = _num(_get(d, "CHLEV"))

    # 标签：1=是 → 阳性；2=否 → 阴性；7/8/9 → 缺失
    label = chol.map({1.0: 1.0, 2.0: 0.0})

    h_cm, w_kg = h * 2.54, w * 0.45359237
    out = pd.DataFrame({
        # 2016-2018 的 AGE_P 上限为 85，2019+ 的 AGEP_A 可达 99；统一封顶保持口径一致
        "age":          _num(_get(d, "AGE_P", "AGEP_A")).clip(upper=85),
        "bmi":          w_kg / (h_cm / 100.0) ** 2,
        "height_cm":    h_cm,
        "weight_kg":    w_kg,
        "smoker":       smk,
        # 代谢综合征组分（与高脂血症同源共病，非其后果）
        "hypertension": _yn(_get(d, "HYPEV", "HYPEV_A")),
        "diabetes":     _yn(_get(d, "DIBEV1", "DIBEV_A")),
        "label":        label,
    })
    return out[out["age"].notna() & (out["age"] >= 18) & out["label"].notna()]


def build_cohort(paths):
    frames = []
    for year, z in sorted(paths.items()):
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as f:
                raw = pd.read_csv(f, low_memory=False)
        h = harmonize(raw, year)
        print(f"  {year}: {len(h):,} 人, 高胆固醇 {int(h['label'].sum()):,} 例")
        frames.append(h)
    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def download(keep_raw=False):
    print("获取 NHIS Sample Adult 公开微数据（2016–2025，共 10 个年度）...")
    paths = download_raw()
    print("\n构建建模队列 ...")
    out = build_cohort(paths)

    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")
    sha = _sha256(OUT_CSV)

    print(f"\n已写出: {OUT_CSV}")
    print(f"形状  : {out.shape[0]:,} 行 × {out.shape[1]} 列")
    print(f"阳性  : {int(out.label.sum()):,} ({out.label.mean():.2%})")
    print(f"SHA256: {sha}")
    print("与构建时逐字节一致" if sha == EXPECTED_CSV_SHA256
          else "注意: 与构建时校验值不同，请以实际训练结果为准")

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
    df = pd.read_csv(OUT_CSV)
    print(f"文件    : {OUT_CSV}")
    print(f"列      : {header}")
    print(f"样本数  : {n:,}   阳性: {int(df.label.sum()):,}")
    print(f"SHA256  : {_sha256(OUT_CSV)}")
    ok = (header == OUTPUT_COLUMNS and n == EXPECTED_N_SAMPLES
          and int(df.label.sum()) == EXPECTED_N_POSITIVE)
    print(f"校验结果: {'通过' if ok else '不通过'}")
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载并构建 NHIS 高脂血症数据集")
    parser.add_argument("--check", action="store_true", help="仅校验已构建的 CSV")
    parser.add_argument("--keep-raw", action="store_true", help="保留下载的原始文件")
    args = parser.parse_args()
    check() if args.check else download(keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
