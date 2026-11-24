"""Budget allocation (§7)."""
import numpy as np

C_SMOOTH = 0.05
W_LO, W_HI = 0.1, 10.0


def normalise(dMIA, dAcc):
    """§7.1: V = max(−ΔMIA, 0), U = max(−ΔAcc, 0), each divided by its mean over the attributes.
    Stops (raises) if either is all zero."""
    V = np.maximum(-np.asarray(dMIA, dtype=np.float64), 0.0)
    U = np.maximum(-np.asarray(dAcc, dtype=np.float64), 0.0)
    if not V.any():
        raise RuntimeError("V = max(−ΔMIA, 0) is all zero; stopping as required by §7.1")
    if not U.any():
        raise RuntimeError("U = max(−ΔAcc, 0) is all zero; stopping as required by §7.1")
    return V / V.mean(), U / U.mean()


def allocate(U, V, c=C_SMOOTH, lo=W_LO, hi=W_HI, iters=200, s_lo=1e-6, s_hi=1e6):
    """§7.2–7.3: raw_j = ((U_j + c)/(V_j + c))^(1/3); find s with Σ clip(s·raw_j, lo, hi) equal to the
    number of attributes, by bisection in log space; w_j = clip(s·raw_j, lo, hi). Returns (w, s)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    total = float(len(U))
    raw = np.cbrt((U + c) / (V + c))

    def w_of(s):
        return np.clip(s * raw, lo, hi)

    a, b = np.log(s_lo), np.log(s_hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if w_of(np.exp(mid)).sum() < total:
            a = mid
        else:
            b = mid
    s = float(np.exp(0.5 * (a + b)))
    w = w_of(s)
    rel = abs(w.sum() - total) / total
    if rel >= 1e-6:
        raise RuntimeError(f"allocate: Σw = {w.sum()} differs from {total} (rel {rel:.2e})")
    return w, s
