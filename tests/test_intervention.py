"""§10.3: FSE-2022 interventions set attribute j of every record to α and touch nothing else."""
import numpy as np
import pytest
import torch

from adaptive_dp.data import load_raw, prepare
from adaptive_dp.defender import CONT_GRID, set_attribute, value_set


@pytest.fixture(scope="module")
def adult():
    X, y, splits, params, _ = prepare(scale=0.2)
    return X[splits["p_def"]], X[splits["z_def"]], params, splits


def test_every_record_gets_alpha_and_other_columns_unchanged(adult):
    Xp, Xz, params, _ = adult
    for a in params["attributes"]:
        cols = a["columns"]
        other = np.setdiff1d(np.arange(Xp.shape[1]), cols)
        for label, vec in value_set(a, Xp):
            for X in (Xp, Xz):
                new = set_attribute(X, cols, vec)
                assert np.array_equal(new[:, cols], np.broadcast_to(vec, (len(X), len(cols))))
                assert np.array_equal(new[:, other], X[:, other])
                if a["type"] == "categorical":
                    assert (new[:, cols].sum(1) == 1).all()
                    pos = new[:, cols].argmax(1)
                    assert (pos == pos[0]).all() and a["categories"][pos[0]] == label


def test_value_set_sizes(adult):
    Xp, _, params, splits = adult
    df, _, _ = load_raw()
    sub = df.iloc[splits["p_def"]]
    for a in params["attributes"]:
        vals = value_set(a, Xp)
        if a["type"] == "categorical":
            present = sub[a["name"]].replace("?", "Unknown").unique()
            assert len(vals) == len(present)                    # every category seen in P_def, once
            assert sorted(v for v, _ in vals) == sorted(present)
        else:
            assert len(vals) == CONT_GRID
            assert np.allclose([v for v, _ in vals], np.linspace(0, 1, CONT_GRID))
            assert len(value_set(a, Xp, n_grid=3)) == 3
            assert np.allclose([v for v, _ in value_set(a, Xp, n_grid=3)], [0, 0.5, 1])


def test_torch_and_numpy_agree(adult):
    Xp, _, params, _ = adult
    for a in params["attributes"]:
        label, vec = value_set(a, Xp)[-1]
        assert np.array_equal(set_attribute(Xp, a["columns"], vec),
                              set_attribute(torch.as_tensor(Xp), a["columns"], vec).numpy())
