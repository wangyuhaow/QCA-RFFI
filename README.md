# QCA-RFFI: Query-Conditioned Same-Packet Amplitude–Phase Alignment for Few-Shot Cross-Receiver Open-Set RF Authentication

Reference implementation and reproducibility artifacts for our IEEE TCCN submission:
> **Query-Conditioned Same-Packet Amplitude–Phase Alignment for Few-Shot Cross-Receiver Open-Set RF Authentication**  
> Yuhao Wang, Junbo Su, Jinhui Cao, Hui Qi, and Xiaoqiang Di (Corresponding Author)

## Method-to-Artifact Name Mapping

| Paper Name | Code Artifact | Description |
|---|---|---|
| **QCA-P** | `proto_50` | Path-A bounded alignment + Path-B query-conditioned prototype verifier |
| **QCA-K** | `pkt_00` | Path-A bounded alignment + Path-B decoupled uniform packetwise verifier |
| **QCA-G** | `conditioned_gaussian` | Zero-shot Path-A weights plugged into shrinkage Gaussian Mahalanobis verifier |
| **MLGPN** | `standalone_mlgpn` | Static shrinkage Gaussian Mahalanobis baseline |
| **Perm-Ctrl** | `permuted_proto50` | Equal-marginal same-context cyclic permutation control ($\Delta^{\mathrm{perm}}$) |
| **Static-Pair** | `explicit_pair` | Equal-weight static alignment control (Layer-1 ablation) |

## Repository Structure (Public vs. Private Policy)

Following standard IEEE physical-layer security reproducibility practice during peer review:
- **Publicly Released Here:**
  - `qca_rffi/descriptor.py` — Exact 48-D same-packet conditional response descriptor $D(x)=[M;\Delta]$ and equal-marginal cyclic permutation control $\Delta^{\mathrm{perm}}$.
  - `qca_rffi/calibration.py` — Algorithm 1 (legitimate-only claim-centered calibration $\lambda=0.75$, zero-extra-calibration `Split-32`, and OSCR/CIAR/FAR/WID metrics).
  - `scripts/verify_prop1_invariance.py` — Standalone numerical verification of Proposition 1 (post-DC complex gain, phase, and linear-CFO rescaling invariance to $10^{-8}\text{--}10^{-7}$).
  - `manifests/` — Internally pre-specified SHA-256 manifest hash (`24e1a1e3...a5c4bca3`) and public WiSig receiver/device split manifest.
- **Kept Private During Review (Available Upon Acceptance / Reasonable Request):**
  - Proprietary pretrained checkpoints (`*.pt`), full end-to-end neural backbone weights, and raw self-acquired BearPi SLE I/Q captures.

## Quick Verification (Proposition 1)

```bash
pip install numpy
python scripts/verify_prop1_invariance.py
```

## Public Datasets Used
- **WiSig ManyTx (802.11a/g OFDM):** Hanna *et al.*, IEEE Access, 2022.
- **ROBUST-6G (866 MHz 2-GFSK):** Ayyıldız *et al.*, EuCNC/6G Summit, 2026.
- **OSU LoRa (915 MHz CSS):** Elmaghbub & Hamdaoui, IEEE Access, 2021.
- **BearPi SLE (2.424 GHz GFSK):** Self-acquired (anonymized sync-field artifacts to be released upon acceptance).
