"""Legitimate-only claim-centered calibration (Algorithm 1) and open-set metrics."""

from __future__ import annotations
import numpy as np


def fit_claim_centered_calibration(
    cal_scores: np.ndarray,
    cal_routes: np.ndarray,
    known_ids: np.ndarray,
    cal_labels: np.ndarray,
    lam: float = 0.75,
    rho: float = 0.95,
) -> tuple[np.ndarray, float]:
    """Fit per-claim median offsets b_k and threshold tau from legitimate calibration data only."""
    known_ids = np.asarray(known_ids)
    K = known_ids.shape[0]
    routed_ids = known_ids[cal_routes]
    correct_mask = routed_ids == cal_labels

    if not np.any(correct_mask):
        return np.zeros(K, dtype=np.float64), float("-inf")

    routed_scores = cal_scores[np.arange(len(cal_routes)), cal_routes]
    e_scores = routed_scores[correct_mask]
    e_labels = cal_labels[correct_mask]
    e_routes = cal_routes[correct_mask]

    m_all = float(np.median(e_scores))
    b_k = np.zeros(K, dtype=np.float64)
    for k_idx, kid in enumerate(known_ids):
        cls_scores = e_scores[e_labels == kid]
        b_k[k_idx] = (float(np.median(cls_scores)) - m_all) if cls_scores.size > 0 else 0.0

    adjusted = e_scores - lam * b_k[e_routes]
    tau = float(np.quantile(adjusted, 1.0 - rho))
    return b_k, tau


def evaluate_open_set(
    scores: np.ndarray,
    routes: np.ndarray,
    known_ids: np.ndarray,
    labels: np.ndarray,
    b_k: np.ndarray,
    tau: float,
    lam: float = 0.75,
) -> dict[str, float]:
    """Compute CIAR, FAR, WID, and closed-set Top-1 at the locked operating point."""
    known_set = set(np.asarray(known_ids).tolist())
    is_legal = np.array([y in known_set for y in labels], dtype=bool)
    is_unk = ~is_legal

    routed_scores = scores[np.arange(len(routes)), routes] - lam * b_k[routes]
    accepted = routed_scores >= tau
    pred_ids = np.asarray(known_ids)[routes]

    lc = int(np.sum(is_legal & accepted & (pred_ids == labels)))
    wid = int(np.sum(is_legal & accepted & (pred_ids != labels)))
    fa = int(np.sum(is_unk & accepted))
    top1 = int(np.sum(is_legal & (pred_ids == labels)))

    n_legal = max(int(np.sum(is_legal)), 1)
    n_unk = max(int(np.sum(is_unk)), 1)
    return {
        "CIAR": 100.0 * lc / n_legal,
        "FAR": 100.0 * fa / n_unk,
        "WID": 100.0 * wid / n_legal,
        "Top1": 100.0 * top1 / n_legal,
    }
