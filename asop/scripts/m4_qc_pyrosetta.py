#!/usr/bin/env python3
"""M4-QC (branch 2): co-fold with Chai-1, then score the interface with PyRosetta.

Same run_batch per-row contract as m4_qc_chai.py:
    python m4_qc_pyrosetta.py input.json      # cwd is this row's folder -> result.json

This is an ORTHOGONAL quality measure to the ipSAE branch: a physics-based
interface energy (ref2015 ΔG_separated, shape complementarity, buried SASA,
H-bonds) rather than a folding-confidence score. The operator picking between this
and the ipSAE branch is choosing a QC *protocol*; see the soundness caveat in
asop/adapters/m4_qc_energy.py.

SELF-CONTAINED (run_batch copies only this file). It drives two installed tools by
subprocess — the Chai launcher and the PyRosetta `rosetta_tool.py interface`
command (A=target, B=binder, repacked to match the repo protocol) — and converts
the Chai CIF to PDB inline. Set ASOP_QC_MOCK=1 to score without either tool.

NOT yet validated on a live Aurora run: confirm `rosetta_tool.py interface` output
keys and the CIF->PDB atom records against a real complex before trusting numbers.
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

# ---- installed tools (paths from env with the skill defaults) ---- #
CHAI_PY = os.environ.get("ASOP_CHAI_PY", "/flare/FRAME-IDP/msinclair/software/chai-lab/.venv/bin/python")
CHAI_LAUNCH = os.environ.get(
    "ASOP_CHAI_LAUNCH",
    "/flare/FRAME-IDP/msinclair/software/chai-lab/skills/structure-prediction/chai/scripts/chai_xpu.py")
CHAI_DOWNLOADS = os.environ.get("ASOP_CHAI_DOWNLOADS", "/flare/FRAME-IDP/msinclair/software/chai-lab/downloads")
ROSETTA_PY = os.environ.get("ASOP_ROSETTA_PY", "/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/python")
ROSETTA_TOOL = os.environ.get(
    "ASOP_ROSETTA_TOOL", "/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/scripts/rosetta_tool.py")


# --------------------------------------------------------------------------- #
# Chai co-fold (compact; duplicated by the self-contained constraint)
# --------------------------------------------------------------------------- #


def write_fasta(path: Path, target_seq: str, binder_seq: str, ligands: list[dict]) -> None:
    blocks = [f">protein|name=target\n{target_seq}", f">protein|name=binder\n{binder_seq}"]
    for lig in ligands:
        blocks.append(f">ligand|name={lig['name']}\n{lig['smiles']}")
    path.write_text("\n".join(blocks) + "\n")


def run_chai(fasta: Path, outdir: Path, *, seed: int, tile: int, num_samples: int) -> None:
    env = {**os.environ, "ZE_AFFINITY_MASK": str(tile), "CHAI_DOWNLOADS_DIR": CHAI_DOWNLOADS}
    cmd = [CHAI_PY, CHAI_LAUNCH, "fold", str(fasta), str(outdir), "--device", "xpu:0",
           "--seed", str(seed), "--num-diffn-samples", str(num_samples)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = (proc.stdout or "") + (proc.stderr or "")
    if "Failed to tokenize" in log:
        raise RuntimeError("Chai failed to tokenize a ligand (cofactor dropped silently)")
    if proc.returncode != 0:
        raise RuntimeError(f"Chai exit {proc.returncode}: {log[-600:]}")


def best_cif(outdir: Path) -> Path:
    npzs = sorted(glob.glob(str(outdir / "scores.model_idx_*.npz")))
    if not npzs:
        raise RuntimeError("Chai produced no scores npz")
    best_i, best_agg = None, -1e9
    for f in npzs:
        agg = float(np.asarray(np.load(f)["aggregate_score"]).mean())
        if agg > best_agg:
            best_agg, best_i = agg, f
    idx = Path(best_i).stem.split("model_idx_")[-1]
    return outdir / f"pred.model_idx_{idx}.cif"


# --------------------------------------------------------------------------- #
# Minimal CIF -> PDB (chai mmCIF _atom_site loop -> fixed-column PDB records)
# --------------------------------------------------------------------------- #

_AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    lines = cif_path.read_text().splitlines()
    cols, rows, i, n = [], [], 0, len(lines)
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
    idx = {c: k for k, c in enumerate(cols)}

    def col(*names):
        for nm in names:
            if nm in idx:
                return idx[nm]
        return None

    gi = col("_atom_site.group_PDB")
    ai = col("_atom_site.label_atom_id")
    ei = col("_atom_site.type_symbol")
    ci = col("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    ri = col("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    ni = col("_atom_site.label_comp_id", "_atom_site.auth_comp_id")
    xi = col("_atom_site.Cartn_x")
    yi = col("_atom_site.Cartn_y")
    zi = col("_atom_site.Cartn_z")
    bi = col("_atom_site.B_iso_or_equiv")
    out = []
    serial = 0
    for r in rows:
        f = r.split()
        need = [x for x in (ai, ci, ri, ni, xi, yi, zi) if x is not None]
        if not need or max(need) >= len(f):
            continue
        serial += 1
        resn = f[ni].strip('"')[:3]
        rec = "ATOM  " if (gi is not None and f[gi] == "ATOM") or resn in _AA3 else "HETATM"
        atom = f[ai].strip('"')
        elem = (f[ei].strip('"') if ei is not None else atom[0])[:2]
        chain = f[ci][:1]
        try:
            resi = int(f[ri])
            x, y, z = float(f[xi]), float(f[yi]), float(f[zi])
        except ValueError:
            continue
        b = float(f[bi]) if bi is not None and f[bi] not in (".", "?") else 0.0
        atom_name = (f" {atom}" if len(atom) < 4 else atom)[:4].ljust(4)
        out.append(
            f"{rec}{serial:5d} {atom_name}{resn:>3s} {chain}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {elem:>2s}")
    pdb_path.write_text("\n".join(out) + "\nEND\n")


# --------------------------------------------------------------------------- #
# PyRosetta interface scoring (via the installed rosetta_tool.py)
# --------------------------------------------------------------------------- #


def rosetta_interface(pdb_path: Path) -> dict:
    out_json = Path("rosetta.json")
    cmd = [ROSETTA_PY, ROSETTA_TOOL, "interface", str(pdb_path), "--partners", "A_B",
           "--pack-input", "--pack-separated", "--json", str(out_json)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rosetta_tool exit {proc.returncode}: {(proc.stderr or '')[-500:]}")
    data = json.loads(out_json.read_text()) if out_json.is_file() else json.loads(proc.stdout)
    # parse defensively: key names vary between the single-struct and batch tools
    def pick(*names, default=0.0):
        for nm in names:
            if nm in data:
                return float(data[nm])
        return default
    return dict(dg_separated=pick("interface_dG", "dG", "interface_delta_dG"),
                dsasa=pick("dSASA_interface", "dSASA", "interface_delta_sasa"),
                sc=pick("shape_complementarity", "sc", "sc_value"),
                hbonds=pick("interface_hbonds", "hbonds"),
                nres_interface=pick("nres_interface", "interface_nres"))


# --------------------------------------------------------------------------- #
# Mock path (offline): deterministic dG/sc from the sequences.
# --------------------------------------------------------------------------- #


def mock_metrics(design_id: str, target_seq: str, binder_seq: str) -> dict:
    h = int(hashlib.sha256(f"{design_id}|{binder_seq}|{target_seq}".encode()).hexdigest(), 16)
    dg = round(-5.0 - (h % 2500) / 100.0, 2)          # deterministic in [-30, -5]
    return dict(dg_separated=dg, dsasa=round(600 + (h % 1200), 1),
                sc=round(0.45 + (h % 400) / 1000.0, 3), hbonds=2 + h % 8,
                nres_interface=10 + h % 15, mock=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

DEFAULT_DG_GATE = -10.0     # REU; more negative is better binding


def main() -> int:
    inp = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "input.json").read_text())
    design_id = str(inp.get("design_id", inp.get("id", "unknown")))
    gate = float(inp.get("dg_gate", DEFAULT_DG_GATE))
    result = {"design_id": design_id,
              "objective_definition": f"fraction of designs with interface ΔG<={gate:.1f} REU"}
    try:
        target_seq, binder_seq = inp["target_sequence"], inp["binder_sequence"]
        if os.environ.get("ASOP_QC_MOCK") == "1":
            m = mock_metrics(design_id, target_seq, binder_seq)
        else:
            write_fasta(Path("complex.fasta"), target_seq, binder_seq, inp.get("ligands", []))
            run_chai(Path("complex.fasta"), Path("chai_out"), seed=int(inp.get("seed", 42)),
                     tile=int(inp.get("tile", 3)), num_samples=int(inp.get("num_samples", 5)))
            pdb = Path("model.pdb")
            cif_to_pdb(best_cif(Path("chai_out")), pdb)
            m = {**rosetta_interface(pdb), "mock": False}
        m["gate_pass"] = int(m["dg_separated"] <= gate)
        result.update(m)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    Path("result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
