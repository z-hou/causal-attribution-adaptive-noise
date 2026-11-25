"""Folktables ACSEmployment (California, ACS 2018 1-Year person PUMS): loading, checks, attributes, split.

Drop-in replacement for adaptive_dp.data.prepare(): returns X (float32), y, splits, params, info with the
same structure, so the Adult pipeline (adaptive_dp.run) runs on ACSEmployment unchanged.

The task is folktables.ACSEmployment exactly (folktables/acs.py):
- the 16 features AGEP, SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, NATIVITY, DEAR, DEYE, DREM, SEX,
  RAC1P; label ESR == 1 ("civilian employed, at work"; a missing ESR counts as 0);
- no row filter (preprocess is the identity); missing values become 0. folktables writes
  np.nan_to_num(x, -1), but there −1 is numpy's `copy` argument, so NaN is replaced by the default 0.0;
  0 is not a valid code of any attribute with missing values (checked), so this is just the N/A category.
Raw data are the files folktables downloads for ACSDataSource(survey_year='2018', horizon='1-Year',
survey='person').get_data(states=['CA']).

Deduplication (added for this experiment): 177,037 of the 378,817 records repeat another record exactly
(same 16 features and label), mostly children. Identical records could fall into both the member and the
non-member set, so only the first occurrence (file order) of every distinct (x, y) is kept, before splitting.

Attribute types follow the 2018 PUMS data dictionary: AGEP is numeric (N) → continuous, min-max scaled with
the min/max of P_def as for Adult; the other 15 are character codes (C) → categorical, with the full
category list from the dictionary (its N/A code 'b'/'bb' is 0), including categories absent from the data.

    python ACSEmployment/acs_data.py     # prints the data checks and preprocessing summary
"""
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from adaptive_dp.data import SPLITS, check_one_hot, groups, make_splits  # noqa: E402

RAW = HERE / "data" / "raw"
ZIP, CSV_IN_ZIP, DICT = RAW / "csv_pca.zip", "psam_p06.csv", RAW / "PUMS_Data_Dictionary_2018.csv"
MD5 = {"csv_pca.zip": "2d87384683b319f0b2f5901d1eda569c",
       "PUMS_Data_Dictionary_2018.csv": "b7bf7eb7632ca40165aba018d5ce969a"}
N_ROWS = 378_817                                  # California 2018 1-Year person records
N_UNIQUE = 201_780                                # distinct (x, y) records
FEATURES = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
            "DEAR", "DEYE", "DREM", "SEX", "RAC1P"]  # folktables.ACSEmployment.features, j = 1…16
LABEL = "ESR"
CONTINUOUS = {"AGEP"}
NA = 0                                            # value of a missing feature (as folktables)


def check_md5(path):
    h = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if h != MD5[Path(path).name]:
        raise RuntimeError(f"{Path(path).name}: md5 {h} != {MD5[Path(path).name]}")


def parse_dictionary(path=DICT):
    """Variable type ('N' or 'C') and, for 'C', the integer codes from the 2018 PUMS data dictionary;
    the N/A code ('b', 'bb') becomes NA = 0, which must not clash with a valid code."""
    check_md5(path)
    kind, codes = {}, {}
    with open(path, newline="", encoding="latin-1") as f:
        for row in csv.reader(f):
            if not row or row[1] not in FEATURES + [LABEL]:
                continue
            name = row[1]
            if row[0] == "NAME":
                kind[name], codes[name] = row[2], []
            elif row[0] == "VAL" and kind.get(name) == "C":
                lo, hi = row[4], row[5]
                if lo != hi:
                    raise RuntimeError(f"{name}: code range {lo}–{hi} for a character variable")
                code = NA if set(lo) == {"b"} else int(lo)
                if code in codes[name]:
                    raise RuntimeError(f"{name}: the N/A value {NA} clashes with a valid code")
                codes[name].append(code)
    missing = set(FEATURES + [LABEL]) - set(kind)
    if missing:
        raise RuntimeError(f"not in the data dictionary: {sorted(missing)}")
    return kind, {n: sorted(c) for n, c in codes.items() if kind[n] == "C"}


def load_raw():
    """Read the 16 features and ESR of psam_p06.csv (folktables dtypes), apply ACSEmployment.
    Returns exactly folktables.ACSEmployment.df_to_numpy (as a DataFrame of ints), before deduplication."""
    check_md5(ZIP)
    with zipfile.ZipFile(ZIP) as z, z.open(CSV_IN_ZIP) as f:
        df = pd.read_csv(f, usecols=FEATURES + [LABEL, "RT"], dtype={"RT": str})
    if len(df) != N_ROWS:
        raise RuntimeError(f"expected {N_ROWS} rows, found {len(df)}")
    if set(df["RT"]) != {"P"}:
        raise RuntimeError("not a person file")
    feats = df[FEATURES].to_numpy(dtype=np.float64)
    if np.isnan(feats[:, 0]).any():
        raise RuntimeError("AGEP has missing values")
    feats = np.nan_to_num(feats, nan=NA)          # folktables postprocess (NaN → 0, see above)
    if not np.array_equal(feats, np.round(feats)):
        raise RuntimeError("non-integer feature values")
    X_raw = pd.DataFrame(feats.astype(np.int64), columns=FEATURES)
    y = (df[LABEL] == 1).to_numpy().astype(np.int64)   # target_transform: ESR == 1 (NaN → False)
    info = {"rows": len(df), "attributes": len(FEATURES)}
    return X_raw, y, info


def deduplicate(X_raw, y):
    """Keep the first occurrence (file order) of every distinct record (16 features + label)."""
    keep = ~pd.concat([X_raw, pd.Series(y, name="y")], axis=1).duplicated(keep="first").to_numpy()
    if keep.sum() != N_UNIQUE:
        raise RuntimeError(f"expected {N_UNIQUE} distinct records, found {keep.sum()}")
    return X_raw[keep].reset_index(drop=True), y[keep], int((~keep).sum())


def fit_preprocess(df, p_def_idx, kind, codes):
    """Min/max of AGEP on P_def only; categorical attributes get the full dictionary category list."""
    params = {"dataset": "acs_employment_CA_2018", "n_classes": 2, "attributes": [], "dropped": [],
              "d_in": 0}
    col = 0
    sub = df.iloc[p_def_idx]
    for j, name in enumerate(FEATURES, start=1):
        if (name in CONTINUOUS) != (kind[name] == "N"):
            raise RuntimeError(f"{name}: dictionary type {kind[name]} does not match")
        entry = {"j": j, "name": name, "type": "continuous" if name in CONTINUOUS else "categorical"}
        if name in CONTINUOUS:
            v = sub[name].to_numpy(dtype=np.float64)
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                raise RuntimeError(f"{name}: constant on P_def, cannot min-max scale")
            entry.update({"min": lo, "max": hi, "columns": [col]})
            col += 1
        else:
            entry.update({"categories": [str(c) for c in codes[name]],
                          "columns": list(range(col, col + len(codes[name])))})
            col += len(codes[name])
        params["attributes"].append(entry)
    params["d_in"] = col
    return params


def encode(df, params):
    X = np.zeros((len(df), params["d_in"]), dtype=np.float32)
    for a in params["attributes"]:
        v = df[a["name"]].to_numpy()
        if a["type"] == "continuous":
            X[:, a["columns"][0]] = np.clip((v - a["min"]) / (a["max"] - a["min"]), 0.0, 1.0)
        else:
            index = {int(c): i for i, c in enumerate(a["categories"])}
            unknown = sorted(set(np.unique(v)) - set(index))
            if unknown:
                raise RuntimeError(f"{a['name']}: codes not in the data dictionary: {unknown}")
            X[np.arange(len(df)), np.asarray(a["columns"])[pd.Series(v).map(index).to_numpy()]] = 1.0
    return X


def prepare(scale=1.0):
    """Full data pipeline. Returns X (float32), y, splits, params, info (as adaptive_dp.data.prepare)."""
    kind, codes = parse_dictionary()
    df, y, info = load_raw()
    df, y, info["duplicates_removed"] = deduplicate(df, y)
    info["rows_unique"], info["positive_rate"] = len(df), float(y.mean())
    splits = make_splits(n_rows=len(df), scale=scale)
    params = fit_preprocess(df, splits["p_def"], kind, codes)
    X = encode(df, params)
    check_one_hot(X, params)
    if len(np.unique(np.column_stack([X, y]), axis=0)) != len(X):
        raise RuntimeError("encoded records are not unique")
    info["d_in"] = params["d_in"]
    info["n_cols"] = [len(c) for c in groups(params)]
    info["absent_categories"] = {a["name"]: [c for c in a["categories"] if int(c) not in set(df[a["name"]])]
                                 for a in params["attributes"] if a["type"] == "categorical"}
    return X, y, splits, params, info


if __name__ == "__main__":
    X, y, splits, params, info = prepare()
    print(f"rows={info['rows']} duplicates removed={info['duplicates_removed']} unique={info['rows_unique']} "
          f"attributes={info['attributes']} positive_rate={info['positive_rate']:.4f} d_in={info['d_in']}")
    print("split sizes:", {k: len(v) for k, v in splits.items()}, "(Adult sizes:", dict(SPLITS), ")")
    for a in params["attributes"]:
        extra = (f"min={a['min']:g} max={a['max']:g}" if a["type"] == "continuous"
                 else f"k={len(a['categories'])} categories={a['categories']}")
        print(f"  j={a['j']:2d} {a['name']:9s} {a['type']:11s} cols={a['columns'][0]}–{a['columns'][-1]} {extra}")
    print("categories absent from the data:", {k: v for k, v in info["absent_categories"].items() if v})
    print("positive rate per split:", {k: round(float(y[v].mean()), 4) for k, v in splits.items()})
