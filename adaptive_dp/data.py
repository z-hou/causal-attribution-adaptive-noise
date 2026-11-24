"""Adult (AGENT.md §4): loading, checks, attribute groups, six-way split."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "adult"

RAW_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "label"]
# j = 1…12 in this order.
ATTRIBUTES = [("age", "continuous"), ("workclass", "categorical"), ("education-num", "continuous"),
              ("marital-status", "categorical"), ("occupation", "categorical"),
              ("relationship", "categorical"), ("race", "categorical"), ("sex", "categorical"),
              ("capital-gain", "continuous"), ("capital-loss", "continuous"),
              ("hours-per-week", "continuous"), ("native-country", "categorical")]
LOG1P = {"capital-gain", "capital-loss"}
DROPPED = ["fnlwgt", "education"]
UNKNOWN = "Unknown"
N_ROWS = 48_842
SPLITS = [("train", 7000), ("test", 7000), ("p_def", 14000), ("z_def", 2000),
          ("p_att", 14000), ("z_att", 2000)]
SPLIT_SEED = 0


def load_raw(raw_dir=RAW):
    """Merge adult.data and adult.test, strip fields, drop the '.' on test labels; run §4.1 checks."""
    raw_dir = Path(raw_dir)
    for f in ("adult.data", "adult.test", "adult.names"):
        if not (raw_dir / f).exists():
            raise RuntimeError(f"missing {raw_dir / f}")
    train = pd.read_csv(raw_dir / "adult.data", header=None, names=RAW_COLUMNS, dtype=str)
    test = pd.read_csv(raw_dir / "adult.test", header=None, names=RAW_COLUMNS, dtype=str, skiprows=1)
    df = pd.concat([train, test], ignore_index=True)
    df = df.apply(lambda s: s.str.strip())
    df["label"] = df["label"].str.rstrip(".")

    if len(df) != N_ROWS:
        raise RuntimeError(f"expected {N_ROWS} rows, found {len(df)}")
    if df.shape[1] != 15 or df.drop(columns="label").shape[1] != 14:
        raise RuntimeError(f"expected 14 attributes + 1 label, found {df.shape[1]} columns")
    if df.isna().any().any():
        raise RuntimeError("empty fields after parsing")
    labels = sorted(df["label"].unique())
    if labels != ["<=50K", ">50K"]:
        raise RuntimeError(f"unexpected labels {labels}")
    y = (df["label"] == ">50K").to_numpy().astype(np.int64)
    info = {"rows": len(df), "attributes": 14, "labels": labels,
            "rows_adult_data": len(train), "rows_adult_test": len(test),
            "positive_rate": float(y.mean())}
    return df, y, info


def parse_names(path=RAW / "adult.names"):
    """Category lists of the categorical attributes from adult.names, plus 'Unknown'."""
    cats = {}
    names = {a for a, t in ATTRIBUTES if t == "categorical"}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        m = re.match(r"^([a-z-]+):\s*(.+)\.\s*$", line.strip())
        if m and m.group(1) in names:
            cats[m.group(1)] = [c.strip() for c in m.group(2).split(",")] + [UNKNOWN]
    if set(cats) != names:
        raise RuntimeError(f"could not parse categories for {sorted(names - set(cats))}")
    return cats


def make_splits(n_rows=N_ROWS, scale=1.0, seed=SPLIT_SEED):
    """Cut the six subsets in order from one seeded permutation (§4.3)."""
    perm = np.random.default_rng(seed).permutation(n_rows)
    out, start = {}, 0
    for name, size in SPLITS:
        size = int(round(size * scale))
        out[name] = perm[start:start + size]
        start += size
    names = list(out)
    for i in range(len(names)):                      # §10.1: pairwise disjoint
        for k in range(i + 1, len(names)):
            assert np.intersect1d(out[names[i]], out[names[k]]).size == 0, (names[i], names[k])
    return out


def fit_preprocess(df, p_def_idx, categories):
    """Min/max of continuous attributes on P_def only (after log1p where required)."""
    params = {"dataset": "adult", "n_classes": 2, "attributes": [], "dropped": list(DROPPED), "d_in": 0}
    col = 0
    sub = df.iloc[p_def_idx]
    for j, (name, kind) in enumerate(ATTRIBUTES, start=1):
        entry = {"j": j, "name": name, "type": kind}
        if kind == "continuous":
            v = sub[name].astype(float).to_numpy()
            if name in LOG1P:
                v = np.log1p(v)
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                raise RuntimeError(f"{name}: constant on P_def, cannot min-max scale")
            entry.update({"log1p": name in LOG1P, "min": lo, "max": hi, "columns": [col]})
            col += 1
        else:
            entry.update({"categories": categories[name],
                          "columns": list(range(col, col + len(categories[name])))})
            col += len(categories[name])
        params["attributes"].append(entry)
    params["d_in"] = col
    return params


def groups(params):
    """G_j: the encoded column indices of attribute j (list of 12 lists)."""
    return [a["columns"] for a in params["attributes"]]


def encode(df, params):
    X = np.zeros((len(df), params["d_in"]), dtype=np.float32)
    for a in params["attributes"]:
        if a["type"] == "continuous":
            v = df[a["name"]].astype(float).to_numpy()
            if a["log1p"]:
                v = np.log1p(v)
            X[:, a["columns"][0]] = np.clip((v - a["min"]) / (a["max"] - a["min"]), 0.0, 1.0)
        else:
            vals = df[a["name"]].replace("?", UNKNOWN)
            index = {c: i for i, c in enumerate(a["categories"])}
            unknown = sorted(set(vals) - set(index))
            if unknown:
                raise RuntimeError(f"{a['name']}: values not in adult.names: {unknown}")
            X[np.arange(len(df)), np.asarray(a["columns"])[vals.map(index).to_numpy()]] = 1.0
    return X


def check_one_hot(X, params):
    """§10.1: every categorical attribute has exactly one 1 in every record."""
    for a in params["attributes"]:
        if a["type"] == "categorical":
            block = X[:, a["columns"]]
            if not (np.isin(block, (0.0, 1.0)).all() and (block.sum(1) == 1).all()):
                raise RuntimeError(f"one-hot check failed for {a['name']}")


def prepare(scale=1.0, out_dir=None):
    """Full data pipeline. Returns X (float32), y, splits, params, info."""
    df, y, info = load_raw()
    cats = parse_names()
    splits = make_splits(scale=scale)
    params = fit_preprocess(df, splits["p_def"], cats)
    X = encode(df, params)
    check_one_hot(X, params)
    _save(splits, params, out_dir)
    info["d_in"] = params["d_in"]
    info["n_cols"] = [len(c) for c in groups(params)]
    return X, y, splits, params, info


def _save(splits, params, out_dir):
    if out_dir is not None:
        np.savez(Path(out_dir) / "splits.npz", **splits)
        (Path(out_dir) / "preprocess.json").write_text(json.dumps(params, indent=2, ensure_ascii=False))
