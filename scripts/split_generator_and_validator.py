"""Lightweight Split Generator & Anti-Leakage Manifest Validator for QCA-RFFI.

Provides:
  1. Validation of JSON split manifests (e.g., manifests/wisig_split_manifest.json):
     - Checks strict mutual exclusivity of fit, known, and unknown transmitter sets;
     - Checks strict disjointness of seen and unseen receiver sets;
     - Verifies non-overlapping packet slice boundaries;
     - Prints a concise split summary and canonical SHA-256 digest.
  2. Deterministic seed-based device/receiver split generator (for reproducing
     disjoint Fit / Known / Unknown partitions and outputting a SHA-256 manifest).

Contains no neural network definitions, training code, model weights, or raw IQ data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def canonical_sha256(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_slice_bounds(desc: str) -> tuple[int, int] | None:
    m = re.search(r"\[(\d+):(\d+)\]", str(desc))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def validate_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    fit_tx = set(data.get("fit_tx", []))
    known_tx = set(data.get("eval_known_tx", []))
    unk_tx = set(data.get("eval_unknown_tx", []))

    assert fit_tx.isdisjoint(known_tx), f"Overlap fit vs known: {fit_tx & known_tx}"
    assert fit_tx.isdisjoint(unk_tx), f"Overlap fit vs unknown: {fit_tx & unk_tx}"
    assert known_tx.isdisjoint(unk_tx), f"Overlap known vs unknown: {known_tx & unk_tx}"

    seen_rx = set(data.get("seen_rx", []))
    unseen_rx = set(data.get("unseen_rx", []))
    assert seen_rx.isdisjoint(unseen_rx), f"Overlap seen vs unseen Rx: {seen_rx & unseen_rx}"

    slices = data.get("day_packet_slices", {})
    eval_keys = ["enrollment", "guard_1", "calibration", "guard_2", "query"]
    intervals = []
    for k in eval_keys:
        if k in slices:
            bounds = parse_slice_bounds(slices[k])
            if bounds is not None:
                intervals.append((k, bounds[0], bounds[1]))
    intervals.sort(key=lambda x: x[1])
    for i in range(len(intervals) - 1):
        name_a, _, end_a = intervals[i]
        name_b, start_b, _ = intervals[i + 1]
        assert end_a <= start_b, f"Packet slice overlap between {name_a} (end={end_a}) and {name_b} (start={start_b})"

    digest = canonical_sha256(data)
    summary = {
        "manifest": str(manifest_path),
        "n_fit_tx": len(fit_tx),
        "n_eval_known_tx": len(known_tx),
        "n_eval_unknown_tx": len(unk_tx),
        "n_seen_rx": len(seen_rx),
        "n_unseen_rx": len(unseen_rx),
        "n_dates": len(data.get("dates", [])),
        "eval_packet_slices": {k: (s, e) for k, s, e in intervals},
        "canonical_sha256": digest,
        "leakage_checks": "PASSED",
    }
    return summary


def generate_disjoint_split(
    seed: int,
    n_total_tx: int = 40,
    n_fit: int = 16,
    n_known: int = 12,
    n_unknown: int = 12,
    seen_rx: tuple[str, ...] = ("18-2", "3-19", "8-14", "8-7"),
    unseen_rx: tuple[str, ...] = ("1-1", "1-19"),
) -> dict:
    assert n_fit + n_known + n_unknown <= n_total_tx, "Split sizes exceed total transmitters."
    rng = np.random.default_rng(seed)
    all_tx = [f"Tx-{i:02d}" for i in range(1, n_total_tx + 1)]
    perm = rng.permutation(all_tx).tolist()

    manifest = {
        "seed": seed,
        "seen_rx": list(seen_rx),
        "unseen_rx": list(unseen_rx),
        "fit_tx": sorted(perm[:n_fit]),
        "eval_known_tx": sorted(perm[n_fit : n_fit + n_known]),
        "eval_unknown_tx": sorted(perm[n_fit + n_known : n_fit + n_known + n_unknown]),
        "day_packet_slices": {
            "fit": "[0:50]",
            "enrollment": "[0:8]",
            "guard_1": "[8:10]",
            "calibration": "[10:26]",
            "guard_2": "[26:28]",
            "query": "[28:50]",
        },
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manifests" / "wisig_split_manifest.json",
        help="Path to an existing JSON split manifest to validate.",
    )
    parser.add_argument(
        "--generate-seed",
        type=int,
        default=None,
        help="Optional random seed to generate a fresh disjoint split and print its SHA-256 manifest.",
    )
    args = parser.parse_args()

    if args.generate_seed is not None:
        gen = generate_disjoint_split(seed=args.generate_seed)
        print(json.dumps(gen, indent=2, ensure_ascii=False))
        return

    summary = validate_manifest(args.validate)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
