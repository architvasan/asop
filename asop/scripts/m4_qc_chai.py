#!/usr/bin/env python3
"""M4-QC: co-fold a target+binder complex with Chai-1 and score the interface.

run_batch per-row contract (loamlite hpc-deployment):
    python m4_qc_chai.py input.json         # cwd is this row's folder
  input.json: {"design_id", "target_sequence", "binder_sequence",
               optional "ligands":[{"name","smiles"}], "ipsae_gate", "seed",
               "tile", "num_samples"}
  writes result.json into the cwd and exits 0. A real failure writes
  result.json with an "error" field (still exit 0) so the harness marks the row
  failed without crashing the batch.

SELF-CONTAINED on purpose: run_batch copies only this one file to the worker, so
there are no local imports. The ipSAE/pDockQ scoring below is vendored verbatim
(numpy-only) from the campaign's prior EGFR worker, itself a faithful port of
msinclair-py/molecular-simulations ipSAE.py (Dunbar & Elofsson 2024, "ipSAE:
interface-restricted pTM"). Keep it in sync with that reference; do not "improve"
the metric here without validating against it.

Chai-1 is driven through the mandatory Aurora launcher (the stock `chai` CLI is
broken in that environment); paths are taken from env with the skill defaults.
Set ASOP_QC_MOCK=1 to score without Chai (deterministic, for offline tests).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# ipSAE scoring (vendored, numpy-only). PAE convention: PAE[i, j] = expected
# error of residue j when the structure is aligned on residue i; ipSAE sums over j.
# --------------------------------------------------------------------------- #


def d0_of(n: int) -> float:
    """TM-score d0 for n valid residues (Yang & Skolnick), floored so it stays real."""
    n = max(int(n), 19)              # (n-15)^(1/3) real and d0>0
    return 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8


def parse_cif_chain_tokens(cif_path: str) -> list[tuple[str, int]]:
    """(chain, resnum) per CA atom in CIF order — aligns to the PAE/pLDDT token axis."""
    toks: list[tuple[str, int]] = []
    lines = Path(cif_path).read_text().splitlines()
    cols: list[str] = []
    rows: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            i += 1
            hdr = []
            while i < n and lines[i].strip().startswith("_atom_site."):
                hdr.append(lines[i].strip())
                i += 1
            if not hdr:
                continue
            cols = hdr
            while i < n:
                ln = lines[i].strip()
                if ln == "" or ln == "loop_" or ln.startswith("_") or ln.startswith("#"):
                    break
                rows.append(ln)
                i += 1
            break
        i += 1
    if not cols:
        return toks
    idx = {c: k for k, c in enumerate(cols)}
    ai = idx.get("_atom_site.label_atom_id")
    ci = idx.get("_atom_site.auth_asym_id", idx.get("_atom_site.label_asym_id"))
    ri = idx.get("_atom_site.auth_seq_id", idx.get("_atom_site.label_seq_id"))
    if ai is None or ci is None or ri is None:
        return toks
    for r in rows:
        f = r.split()
        if max(ai, ci, ri) >= len(f):
            continue
        if f[ai].strip('"') != "CA":
            continue
        try:
            toks.append((f[ci], int(f[ri])))
        except ValueError:
            continue
    return toks


def ipsae_directional(pae: np.ndarray, a_idx: list[int], b_idx: list[int], pae_cut: float) -> float:
    """ipSAE for aligned chain-A residues scoring chain-B residues."""
    if not a_idx or not b_idx:
        return 0.0
    best = 0.0
    for i in a_idx:
        row = pae[i, b_idx]
        valid = row[row < pae_cut]
        if valid.size == 0:
            continue
        d0 = d0_of(valid.size)
        ptm = 1.0 / (1.0 + (valid / d0) ** 2)
        best = max(best, float(ptm.mean()))
    return round(best, 4)


def score_interface(cif_path: str, pae: np.ndarray, plddt: np.ndarray | None,
                    binder_chain: str, target_chain: str, pae_cut: float = 15.0) -> dict:
    toks = parse_cif_chain_tokens(cif_path)
    N = pae.shape[0]
    if len(toks) != N:                   # ligand/atom tokens shift the count; truncate to min
        toks = toks[: min(len(toks), N)]
    a_idx = [k for k, (ch, _) in enumerate(toks) if ch == binder_chain]
    b_idx = [k for k, (ch, _) in enumerate(toks) if ch == target_chain]
    ab = ipsae_directional(pae, a_idx, b_idx, pae_cut)
    ba = ipsae_directional(pae, b_idx, a_idx, pae_cut)
    if_b, if_t = set(), set()
    for i in a_idx:
        for j in b_idx:
            if pae[i, j] < pae_cut or pae[j, i] < pae_cut:
                if_b.add(i)
                if_t.add(j)
    if_plddt = 0.0
    if plddt is not None and (if_b or if_t):
        vals = [float(plddt[k]) for k in (if_b | if_t) if k < plddt.size]
        if vals:
            if_plddt = float(np.mean(vals))
            if if_plddt <= 1.0:
                if_plddt *= 100.0
    return dict(ipsae=max(ab, ba), ipsae_ab=ab, ipsae_ba=ba,
                n_if_binder=len(if_b), n_if_target=len(if_t),
                if_plddt=round(if_plddt, 1), n_tokens=int(N))


# --------------------------------------------------------------------------- #
# Chai-1 co-folding
# --------------------------------------------------------------------------- #

CHAI_PY = os.environ.get(
    "ASOP_CHAI_PY", "/flare/FRAME-IDP/msinclair/software/chai-lab/.venv/bin/python")
# Under run_batch the launcher must be the worker-visible copy; the adapter sets
# ASOP_CHAI_LAUNCH to $HPC_LOAM_HOME/skills/structure-prediction/chai/scripts/chai_xpu.py.
CHAI_LAUNCH = os.environ.get(
    "ASOP_CHAI_LAUNCH",
    "/flare/FRAME-IDP/msinclair/software/chai-lab/skills/structure-prediction/chai/scripts/chai_xpu.py")
CHAI_DOWNLOADS = os.environ.get(
    "ASOP_CHAI_DOWNLOADS", "/flare/FRAME-IDP/msinclair/software/chai-lab/downloads")


def write_fasta(path: Path, target_seq: str, binder_seq: str, ligands: list[dict]) -> None:
    """Chains in input order: A=target, B=binder, then one ligand entity each."""
    blocks = [f">protein|name=target\n{target_seq}",
              f">protein|name=binder\n{binder_seq}"]
    for lig in ligands:
        blocks.append(f">ligand|name={lig['name']}\n{lig['smiles']}")
    path.write_text("\n".join(blocks) + "\n")


def run_chai(fasta: Path, outdir: Path, *, seed: int, tile: int, num_samples: int) -> None:
    outdir.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "ZE_AFFINITY_MASK": str(tile), "CHAI_DOWNLOADS_DIR": CHAI_DOWNLOADS}
    cmd = [CHAI_PY, CHAI_LAUNCH, "fold", str(fasta), str(outdir), "--device", "xpu:0",
           "--seed", str(seed), "--full-data", "--num-diffn-samples", str(num_samples)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = (proc.stdout or "") + (proc.stderr or "")
    if "Failed to tokenize" in log:      # the silent ligand-dropped failure the skill warns about
        raise RuntimeError("Chai failed to tokenize a ligand (cofactor dropped silently)")
    if proc.returncode != 0:
        raise RuntimeError(f"Chai exit {proc.returncode}: {log[-600:]}")


def best_sample(outdir: Path) -> tuple[Path, dict]:
    """Pick the diffusion sample with the highest aggregate_score; return its CIF
    and a dict of its scalar scores (iptm, ptm, aggregate_score)."""
    npzs = sorted(glob.glob(str(outdir / "scores.model_idx_*.npz")))
    if not npzs:
        raise RuntimeError("Chai produced no scores npz")
    best_i, best_agg, best_scores = None, -1e9, None
    for f in npzs:
        d = np.load(f)
        agg = float(np.asarray(d["aggregate_score"]).mean())
        if agg > best_agg:
            best_agg, best_i, best_scores = agg, f, d
    idx = Path(best_i).stem.split("model_idx_")[-1]
    cif = outdir / f"pred.model_idx_{idx}.cif"
    scalars = {k: float(np.asarray(best_scores[k]).mean())
               for k in ("iptm", "ptm", "aggregate_score") if k in best_scores.files}
    scalars["_npz"] = best_i
    scalars["_cif"] = str(cif)
    return cif, scalars


# --------------------------------------------------------------------------- #
# Mock path (offline tests): deterministic metrics from the sequences, no Chai.
# --------------------------------------------------------------------------- #


def mock_metrics(design_id: str, target_seq: str, binder_seq: str) -> dict:
    h = int(hashlib.sha256(f"{design_id}|{binder_seq}|{target_seq}".encode()).hexdigest(), 16)
    ipsae = round(0.20 + (h % 700) / 1000.0, 4)       # deterministic in [0.20, 0.899]
    return dict(ipsae=ipsae, ipsae_ab=ipsae, ipsae_ba=round(ipsae * 0.9, 4),
                iptm=round(min(1.0, ipsae + 0.05), 4), ptm=round(0.6 + (h % 300) / 1000.0, 4),
                if_plddt=round(60 + (h % 350) / 10.0, 1), n_if_target=8 + h % 12,
                n_tokens=len(target_seq) + len(binder_seq), mock=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    inp = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "input.json").read_text())
    design_id = str(inp.get("design_id", inp.get("id", "unknown")))
    gate = float(inp.get("ipsae_gate", 0.30))
    result = {"design_id": design_id,
              "objective_definition": f"fraction of designs with ipSAE>={gate:.2f}"}
    try:
        target_seq = inp["target_sequence"]
        binder_seq = inp["binder_sequence"]
        if os.environ.get("ASOP_QC_MOCK") == "1":
            m = mock_metrics(design_id, target_seq, binder_seq)
        else:
            ligands = inp.get("ligands", [])
            fasta = Path("complex.fasta")
            outdir = Path("chai_out")
            write_fasta(fasta, target_seq, binder_seq, ligands)
            run_chai(fasta, outdir, seed=int(inp.get("seed", 42)),
                     tile=int(inp.get("tile", 3)), num_samples=int(inp.get("num_samples", 5)))
            cif, scalars = best_sample(outdir)
            d = np.load(scalars["_npz"])
            pae = np.asarray(d["pae"], dtype=float)
            plddt = np.asarray(d["plddt"], dtype=float).reshape(-1) if "plddt" in d.files else None
            iface = score_interface(str(cif), pae, plddt, binder_chain="B", target_chain="A")
            m = {**iface, "iptm": round(scalars.get("iptm", 0.0), 4),
                 "ptm": round(scalars.get("ptm", 0.0), 4),
                 "aggregate_score": round(scalars.get("aggregate_score", 0.0), 4),
                 "best_model": scalars["_cif"], "mock": False}
        m["gate_pass"] = int(m["ipsae"] >= gate)
        result.update(m)
    except Exception as exc:              # a real failure is a failed row, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    Path("result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
