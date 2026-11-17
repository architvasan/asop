"""Tests for the M4-QC script (ipSAE scoring) and its registry adapter.

Runs fully offline: the ipSAE math is exercised on synthetic PAE/CIF, and the
adapter is run through LocalBatchRunner in mock mode (ASOP_QC_MOCK=1), so no Chai
or HPC is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load the self-contained worker as a module (it is designed to run as a script)
_spec = importlib.util.spec_from_file_location("m4_qc_chai", ROOT / "asop" / "scripts" / "m4_qc_chai.py")
qc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc)

from asop.adapters import m4_qc
from asop.batch import LocalBatchRunner
from asop.dag import Executor, RunContext
from asop.registry import Registry
from asop.schemas import Design, DesignBatch, QCReport, TargetSpec
from asop.store import Store


# --------------------------------------------------------------------------- #
# ipSAE math
# --------------------------------------------------------------------------- #


def test_d0_monotonic_and_floored():
    assert qc.d0_of(5) == qc.d0_of(19)          # floored below 19
    assert qc.d0_of(100) > qc.d0_of(30) > qc.d0_of(19)


def test_ipsae_directional_low_pae_beats_high():
    # chains A = tokens [0..19], B = [20..39]
    a_idx, b_idx = list(range(20)), list(range(20, 40))
    low = np.full((40, 40), 0.5)                # confident everywhere
    high = low.copy()
    for i in a_idx:                             # make the A<->B block uncertain beyond the cut
        for j in b_idx:
            high[i, j] = high[j, i] = 25.0
    ip_low = qc.ipsae_directional(low, a_idx, b_idx, pae_cut=15.0)
    ip_high = qc.ipsae_directional(high, a_idx, b_idx, pae_cut=15.0)
    assert 0.0 < ip_low <= 1.0
    assert ip_high == 0.0                       # no interface pair under the cut
    assert ip_low > ip_high


def test_ipsae_directional_empty_chains():
    pae = np.full((4, 4), 1.0)
    assert qc.ipsae_directional(pae, [], [0, 1], 15.0) == 0.0


def _synthetic_cif(path: Path, n_a: int, n_b: int) -> None:
    lines = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.id",
             "_atom_site.label_atom_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"]
    aid = 1
    for ch, n in (("A", n_a), ("B", n_b)):
        for r in range(1, n + 1):
            lines.append(f"ATOM {aid} CA {ch} {r}")
            aid += 1
    path.write_text("\n".join(lines) + "\n#\n")


def test_score_interface_on_synthetic(tmp_path):
    cif = tmp_path / "c.cif"
    _synthetic_cif(cif, 20, 20)
    pae = np.full((40, 40), 1.0)
    plddt = np.full(40, 90.0)
    toks = qc.parse_cif_chain_tokens(str(cif))
    assert len(toks) == 40 and toks[0][0] == "A" and toks[-1][0] == "B"
    out = qc.score_interface(str(cif), pae, plddt, binder_chain="B", target_chain="A")
    assert 0.0 < out["ipsae"] <= 1.0
    assert out["n_if_target"] == 20 and out["n_if_binder"] == 20
    assert out["if_plddt"] == 90.0


def test_mock_metrics_deterministic():
    a = qc.mock_metrics("d1", "MKQL", "GSAA")
    b = qc.mock_metrics("d1", "MKQL", "GSAA")
    assert a == b and 0.20 <= a["ipsae"] < 0.90


# --------------------------------------------------------------------------- #
# adapter end-to-end (mock execution)
# --------------------------------------------------------------------------- #


def test_cofactor_ligands():
    ligs = m4_qc.cofactor_ligands(["FE2", "AKG", "UNKNOWN"])
    names = {l["name"] for l in ligs}
    assert names == {"FE2", "AKG"}              # unknown code skipped, not guessed


def test_m4_qc_adapter_mock(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    runner = LocalBatchRunner(workdir=tmp_path / "b", env={"ASOP_QC_MOCK": "1"})
    registry = Registry(tmp_path / "reg.jsonl", adapters={"m4-qc": m4_qc.ADAPTER})
    registry.register(m4_qc.baseline_record())
    sop = registry.build_sop("v1", ["qc"])

    target = TargetSpec(name="ALKBH1", sequence="MKQLEDKVEEL", cofactors=["FE2", "AKG"])
    batch = DesignBatch(target=target, designs=[
        Design(design_id=f"d{i}", sequence="GSAAEELK" + "A" * i, method="rfd") for i in range(5)])

    report, trace = Executor().run(sop, batch, RunContext(extra={"store": store, "runner": runner}))
    assert isinstance(report, QCReport)
    assert 0.0 <= report.objective <= 1.0
    assert len(report.survivors) == sum(1 for m in report.metrics
                                        if m.name == "ipsae" and m.gate_pass)
    assert len(store.metrics_for("ALKBH1", "ipsae")) == batch.size
    assert "ipSAE" in report.objective_definition


# --------------------------------------------------------------------------- #
# second branch (PyRosetta interface ΔG) + the 2-branch fork
# --------------------------------------------------------------------------- #

_specp = importlib.util.spec_from_file_location(
    "m4_qc_pyrosetta", ROOT / "asop" / "scripts" / "m4_qc_pyrosetta.py")
qcp = importlib.util.module_from_spec(_specp)
_specp.loader.exec_module(qcp)

from asop.adapters import m4_qc_energy
from asop.operator import BanditOperator


def test_pyrosetta_mock_metrics_deterministic_and_gated():
    m = qcp.mock_metrics("d1", "MKQL", "GSAA")
    assert m == qcp.mock_metrics("d1", "MKQL", "GSAA")
    assert -30.0 <= m["dg_separated"] <= -5.0 and 0.0 < m["sc"] < 1.0


def test_cif_to_pdb_roundtrips_chain_and_coords(tmp_path):
    cif = tmp_path / "c.cif"
    cif.write_text(
        "data_t\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n"
        "ATOM 1 N ALA A 1 1.000 2.000 3.000 50.0\n"
        "ATOM 2 CA ALA A 1 1.500 2.500 3.500 50.0\n"
        "ATOM 3 CA GLY B 1 4.000 5.000 6.000 60.0\n#\n")
    pdb = tmp_path / "c.pdb"
    qcp.cif_to_pdb(cif, pdb)
    text = pdb.read_text()
    assert text.count("ATOM  ") == 3
    assert " A   1" in text and " B   1" in text          # both chains present
    assert "1.500   2.500   3.500" in text                # CA coords preserved


def test_energy_adapter_mock_and_fork_selection(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    runner = LocalBatchRunner(workdir=tmp_path / "b", env={"ASOP_QC_MOCK": "1"})
    registry = Registry(tmp_path / "reg.jsonl",
                        adapters={"m4-qc": m4_qc.ADAPTER, "m4-qc-energy": m4_qc_energy.ADAPTER})
    registry.register(m4_qc.baseline_record())       # chai-ipsae (baseline)
    registry.register(m4_qc_energy.record())          # pyrosetta-dG (alternative)

    sop = registry.build_sop("v2", ["qc"])
    fork = sop.steps[0]
    assert set(fork.branches) == {"chai-ipsae", "pyrosetta-dG"}
    assert fork.allow_operator and fork.default_branch == "chai-ipsae"

    target = TargetSpec(name="ALKBH1", sequence="MKQLEDKVEEL", cofactors=["FE2", "AKG"])
    op = BanditOperator(epsilon=0.3, seed=0)
    seen = set()
    for r in range(6):
        batch = DesignBatch(target=target, designs=[
            Design(design_id=f"r{r}-d{i}", sequence="GSAAEELK" + "A" * i, method="rfd") for i in range(5)])
        report, trace = Executor(select=op.select).run(
            sop, batch, RunContext(round=r, extra={"store": store, "runner": runner}))
        op.record("qc", trace[0].branch, report.objective)
        seen.add(trace[0].branch)
        assert isinstance(report, QCReport) and 0.0 <= report.objective <= 1.0
    assert seen == {"chai-ipsae", "pyrosetta-dG"}      # operator explored both branches
    # energy branch wrote dg_separated metrics to the store
    assert store.metrics_for("ALKBH1", "dg_separated")


def test_failed_row_is_not_a_survivor(tmp_path):
    # a row with no sequences makes the worker write an error; it must count as 0
    store = Store(tmp_path / "db.sqlite")
    runner = LocalBatchRunner(workdir=tmp_path / "b", env={"ASOP_QC_MOCK": "1"})
    script = (ROOT / "asop" / "scripts" / "m4_qc_chai.py").read_text()
    from asop.batch import BatchReceipt
    receipt = runner.submit(step="qc", script=script,
                            rows=[{"design_id": "bad"}])        # missing sequences -> error row
    assert isinstance(receipt, BatchReceipt)
    assert receipt.rows[0].get("error")
    assert receipt.ok is False
