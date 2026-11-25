"""Cardio (Kaggle "Cardiovascular Disease dataset", OpenML 45547): loading, checks, attribute groups, split.

Drop-in replacement for adaptive_dp.data.prepare(): returns X (float32), y, splits, params, info with the
same structure, so the Adult pipeline (adaptive_dp.run) runs on Cardio unchanged.

- 70,000 records, 11 attributes + binary label `cardio`; the row id `id` is dropped.
- Continuous (5): age (days), height, weight, ap_hi, ap_lo. height/weight/ap_hi/ap_lo contain obvious
  entry errors (e.g. ap_hi = 16020, ap_lo = −70 or 1000+), so they are first clipped to fixed physiological
  ranges (CLIP), then min-max scaled with the min/max of P_def, as for Adult.
- Categorical (6): gender, cholesterol, gluc, smoke, alco, active; categories as declared in the ARFF header
  (no missing values, so no 'Unknown').
- Six-way split: adaptive_dp.data.make_splits with the Adult sizes and seed (7000 / 7000 / 14000 / 2000 /
  14000 / 2000, seed 0) on the 70,000 rows.

    python Cardio/cardio_data.py      # prints the data checks and preprocessing summary
"""
import hashlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from adaptive_dp.data import SPLITS, check_one_hot, groups, make_splits  # noqa: E402

RAW = HERE / "data" / "raw" / "Cardiovascular-Disease-dataset.arff"
MD5 = "b6b56c8069c390882b66dec870daffa3"        # OpenML file 22116515
N_ROWS = 70_000
LABEL = "cardio"
ROW_ID = "id"
# j = 1…11 in this order (the column order of the file).
ATTRIBUTES = [("age", "continuous"), ("gender", "categorical"), ("height", "continuous"),
              ("weight", "continuous"), ("ap_hi", "continuous"), ("ap_lo", "continuous"),
              ("cholesterol", "categorical"), ("gluc", "categorical"), ("smoke", "categorical"),
              ("alco", "categorical"), ("active", "categorical")]
# Fixed clipping ranges (cm, kg, mmHg) applied before min-max scaling.
CLIP = {"height": (100.0, 220.0), "weight": (30.0, 200.0), "ap_hi": (60.0, 250.0), "ap_lo": (30.0, 150.0)}


def read_arff(path=RAW):
    """Header (attribute name → nominal categories or None) and the data as strings; md5 checked."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != MD5:
        raise RuntimeError(f"{path.name}: md5 {hashlib.md5(raw).hexdigest()} != {MD5}")
    lines = raw.decode("utf-8").splitlines()
    header, start = {}, None
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^@attribute\s+(\S+)\s+(.+)$", s, flags=re.I)
        if m:
            kind = m.group(2).strip()
            header[m.group(1)] = ([c.strip() for c in kind[1:-1].split(",")]
                                  if kind.startswith("{") else None)
        elif s.lower() == "@data":
            start = i + 1
            break
    if start is None:
        raise RuntimeError("no @DATA section")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None, names=list(header), dtype=str)
    return header, df.apply(lambda s: s.str.strip())


def load_raw(path=RAW):
    """Checks: 70,000 rows, id + 11 attributes + label, no missing values, labels {0, 1}."""
    header, df = read_arff(path)
    expected = [ROW_ID] + [a for a, _ in ATTRIBUTES] + [LABEL]
    if list(df.columns) != expected:
        raise RuntimeError(f"unexpected columns {list(df.columns)}")
    if len(df) != N_ROWS:
        raise RuntimeError(f"expected {N_ROWS} rows, found {len(df)}")
    if df.isna().any().any():
        raise RuntimeError("empty fields after parsing")
    if not df[ROW_ID].is_unique:
        raise RuntimeError("duplicate ids")
    labels = sorted(df[LABEL].unique())
    if labels != ["0", "1"]:
        raise RuntimeError(f"unexpected labels {labels}")
    cats = {a: header[a] for a, t in ATTRIBUTES if t == "categorical"}
    if any(c is None for c in cats.values()):
        raise RuntimeError("a categorical attribute is not declared nominal in the ARFF header")
    df = df.drop(columns=ROW_ID)
    y = (df[LABEL] == "1").to_numpy().astype(np.int64)
    info = {"rows": len(df), "attributes": len(ATTRIBUTES), "labels": labels, "positive_rate": float(y.mean())}
    return df, y, cats, info


def continuous_values(df, name):
    v = df[name].astype(float).to_numpy()
    if name in CLIP:
        v = np.clip(v, *CLIP[name])
    return v


def fit_preprocess(df, p_def_idx, categories):
    """Min/max of continuous attributes on P_def only (after clipping where required)."""
    params = {"dataset": "cardio", "n_classes": 2, "attributes": [], "dropped": [ROW_ID], "d_in": 0}
    col = 0
    sub = df.iloc[p_def_idx]
    for j, (name, kind) in enumerate(ATTRIBUTES, start=1):
        entry = {"j": j, "name": name, "type": kind}
        if kind == "continuous":
            v = continuous_values(sub, name)
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                raise RuntimeError(f"{name}: constant on P_def, cannot min-max scale")
            entry.update({"clip": list(CLIP[name]) if name in CLIP else None, "min": lo, "max": hi,
                          "columns": [col]})
            col += 1
        else:
            entry.update({"categories": categories[name],
                          "columns": list(range(col, col + len(categories[name])))})
            col += len(categories[name])
        params["attributes"].append(entry)
    params["d_in"] = col
    return params


def encode(df, params):
    X = np.zeros((len(df), params["d_in"]), dtype=np.float32)
    for a in params["attributes"]:
        if a["type"] == "continuous":
            v = continuous_values(df, a["name"])
            X[:, a["columns"][0]] = np.clip((v - a["min"]) / (a["max"] - a["min"]), 0.0, 1.0)
        else:
            vals = df[a["name"]]
            index = {c: i for i, c in enumerate(a["categories"])}
            unknown = sorted(set(vals) - set(index))
            if unknown:
                raise RuntimeError(f"{a['name']}: values not declared in the ARFF header: {unknown}")
            X[np.arange(len(df)), np.asarray(a["columns"])[vals.map(index).to_numpy()]] = 1.0
    return X


def prepare(scale=1.0):
    """Full data pipeline. Returns X (float32), y, splits, params, info (as adaptive_dp.data.prepare)."""
    df, y, cats, info = load_raw()
    splits = make_splits(n_rows=N_ROWS, scale=scale)
    params = fit_preprocess(df, splits["p_def"], cats)
    X = encode(df, params)
    check_one_hot(X, params)
    info["d_in"] = params["d_in"]
    info["n_cols"] = [len(c) for c in groups(params)]
    info["clipped"] = {n: int(((df[n].astype(float) < lo) | (df[n].astype(float) > hi)).sum())
                       for n, (lo, hi) in CLIP.items()}
    return X, y, splits, params, info


if __name__ == "__main__":
    X, y, splits, params, info = prepare()
    print(f"rows={info['rows']} attributes={info['attributes']} positive_rate={info['positive_rate']:.4f} "
          f"d_in={info['d_in']}")
    print("split sizes:", {k: len(v) for k, v in splits.items()}, "(Adult sizes:", dict(SPLITS), ")")
    print("records clipped (all 70,000):", info["clipped"])
    for a in params["attributes"]:
        extra = (f"clip={a['clip']} min={a['min']:g} max={a['max']:g}" if a["type"] == "continuous"
                 else f"categories={a['categories']}")
        print(f"  j={a['j']:2d} {a['name']:12s} {a['type']:11s} cols={a['columns']} {extra}")
    print(f"positive rate per split:", {k: round(float(y[v].mean()), 4) for k, v in splits.items()})
