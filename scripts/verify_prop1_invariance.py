"""Standalone numerical verification of Proposition 1 (Post-DC Rescaling Invariance)."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qca_rffi.descriptor import extract_descriptor


def main() -> None:
    rng = np.random.default_rng(42)
    L = 256
    lags = (1, 4, 8, 16)
    ctx = {lag: [np.arange(i, min(i + 16, L)) for i in range(lag, L, 16)] for lag in lags}

    diffs, perm_m_diffs = [], []
    for _ in range(200):
        x = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * (1.0 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, L)))
        b = 10.0 ** rng.uniform(-1.0, 1.0)
        alpha = rng.uniform(-np.pi, np.pi)
        omega = rng.uniform(-0.15, 0.15)
        n = np.arange(L, dtype=np.float64)
        x_prime = b * np.exp(1j * (alpha + omega * n)) * x

        M1, _, D1 = extract_descriptor(x, ctx, lags=lags, permute=False)
        _, _, D2 = extract_descriptor(x_prime, ctx, lags=lags, permute=False)
        Mp, _, _ = extract_descriptor(x, ctx, lags=lags, permute=True)
        diffs.append(np.max(np.abs(D1 - D2)))
        perm_m_diffs.append(np.max(np.abs(M1 - Mp)))

    print(f"[Prop. 1] Mean max |D(x') - D(x)| over 200 trials: {np.mean(diffs):.3e} (max: {np.max(diffs):.3e})")
    print(f"[Control] Mean max |M^perm - M| (exact marginal preservation): {np.mean(perm_m_diffs):.3e}")


if __name__ == "__main__":
    main()
