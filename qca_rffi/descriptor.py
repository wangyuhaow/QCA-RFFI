"""Same-packet amplitude-phase conditional response descriptor D(x) = [M; Delta]
and equal-marginal same-context cyclic permutation control Delta^perm.

Reference implementation for:
  "Query-Conditioned Same-Packet Amplitude-Phase Alignment for Few-Shot
   Cross-Receiver Open-Set RF Authentication" (IEEE TCCN submission).
"""

from __future__ import annotations
import numpy as np


def extract_descriptor(
    x: np.ndarray,
    context_groups: dict[int, list[np.ndarray]],
    lags: tuple[int, ...] = (1, 4, 8, 16),
    rms_ratio: float = 0.05,
    derot_floor: float = 0.05,
    permute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract M (24-D), Delta (24-D), and D=[M; Delta] (48-D) from a post-DC burst x.

    Args:
        x: Complex 1-D post-DC baseband synchronization burst of length L.
        context_groups: Mapping lag -> list of 1-D integer index arrays defining
            deterministic protocol-specific context groups g in G_l (indices >= lag).
        lags: Short-lag set L (default: (1, 4, 8, 16)).
        rms_ratio: Relative-RMS validity mask threshold (default: 0.05).
        derot_floor: Denominator stability floor max(|u_bar_l|, derot_floor).
        permute: If True, applies fixed-point-free cyclic permutation inside each
            active context group to produce the equal-marginal control Delta^perm.

    Returns:
        (M, Delta, D) as float64 arrays of shapes (24,), (24,), and (48,).
    """
    x = np.asarray(x, dtype=np.complex128)
    L = x.shape[0]
    centers = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    M_parts, D_parts = [], []

    for lag in lags:
        p = x[lag:] * np.conj(x[:-lag])
        a = np.abs(p)
        thresh = rms_ratio * np.sqrt(np.mean(a * a))
        valid = a >= thresh

        active_groups = []
        active_idx = []
        for g in context_groups[lag]:
            g_arr = np.asarray(g, dtype=np.int64) - lag
            g_arr = g_arr[(g_arr >= 0) & (g_arr < L - lag)]
            g_valid = g_arr[valid[g_arr]]
            if g_valid.size >= 2:
                active_groups.append(g_valid)
                active_idx.append(g_valid)

        if not active_idx:
            M_parts.append(np.zeros(6, dtype=np.float64))
            D_parts.append(np.zeros(6, dtype=np.float64))
            continue

        I_l = np.concatenate(active_idx)
        N_l = I_l.size
        u = p[I_l] / np.maximum(a[I_l], 1e-12)
        u_bar = np.mean(u)
        denom = max(float(np.abs(u_bar)), derot_floor)
        v_active = (u * np.conj(u_bar) - (np.abs(u_bar) ** 2)) / denom

        v_full = np.zeros(L - lag, dtype=np.complex128)
        v_full[I_l] = v_active

        order = np.argsort(a[I_l], kind="mergesort")
        ranks = np.empty(N_l, dtype=np.float64)
        ranks[order] = np.arange(N_l, dtype=np.float64) / max(N_l - 1, 1)

        w_full = np.zeros((3, L - lag), dtype=np.float64)
        for b, c_b in enumerate(centers):
            w_full[b, I_l] = np.maximum(0.0, 1.0 - 2.0 * np.abs(ranks - c_b))

        if permute:
            w_use = w_full.copy()
            for g_valid in active_groups:
                w_use[:, g_valid] = np.roll(w_full[:, g_valid], shift=-1, axis=1)
        else:
            w_use = w_full

        m_lag = np.zeros(6, dtype=np.float64)
        d_lag = np.zeros(6, dtype=np.float64)
        for b in range(3):
            w_sum = max(float(np.sum(w_full[b, I_l])), 1e-12)
            C_lb = np.sum(w_use[b, I_l] * v_full[I_l]) / w_sum
            M_num = 0.0j
            for g_valid in active_groups:
                w_bar_g = float(np.mean(w_full[b, g_valid]))
                M_num += w_bar_g * np.sum(v_full[g_valid])
            M_lb = M_num / w_sum
            Delta_lb = C_lb - M_lb
            m_lag[2 * b : 2 * b + 2] = [M_lb.real, M_lb.imag]
            d_lag[2 * b : 2 * b + 2] = [Delta_lb.real, Delta_lb.imag]

        M_parts.append(m_lag)
        D_parts.append(d_lag)

    M = np.concatenate(M_parts)
    Delta = np.concatenate(D_parts)
    return M, Delta, np.concatenate([M, Delta])
