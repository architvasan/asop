"""M4-QC branch 2: Chai-1 co-fold + PyRosetta interface ΔG (physics-based).

A second, interchangeable branch at the `qc` fork — same `DesignBatch -> QCReport`
contract as the ipSAE branch, so the operator may choose between them. It wraps
the self-contained worker `asop/scripts/m4_qc_pyrosetta.py` and reuses the holo
cofactor mapping and the shared build/parse glue.

SOUNDNESS CAVEAT — read before trusting an operator choice at this fork.
The operator ranks branches by `QCReport.objective` = survivor fraction under each
branch's gate. That is a clean signal when branches *produce* things (more good
binders is unambiguously better), but these two branches are *evaluators* of the
same designs under different gates, so a looser gate reports a higher yield
without being better science. Comparing them by self-reported yield therefore
selects for the more permissive evaluator, not the more predictive one. Two sound
ways to use this fork:
  (a) treat the branches as same-metric implementations to trade off on cost/speed
      (e.g. Chai-ipSAE vs Boltz-ipSAE), not orthogonal metrics; or
  (b) make the objective a GROUND signal both branches are scored against — e.g.
      agreement with a held-out reference, or downstream (MD / lab) confirmation —
      rather than either branch's own pass count.
Until (b) lands, prefer orthogonal filters (ipSAE AND ΔG AND developability) as
SEQUENTIAL funnel steps over an operator choosing one evaluator by its own yield.
This branch is provided so the fork machinery is real and A/B-able; the objective
wiring is the open item (TODO: bind a ground signal into `QCReport.objective`).
"""

from __future__ import annotations

from pathlib import Path

from ..dag import Node
from ..nodes import batch_node
from ..registry import Adapter, BranchRecord
from ..schemas import DesignBatch, QCReport
from ..store import Store
from . import _common
from .m4_qc import cofactor_ligands

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "m4_qc_pyrosetta.py"


def _build_rows(store: Store, inp: DesignBatch):
    return _common.make_build_rows()(store, inp, cofactor_ligands(inp.target.cofactors))


def make_node(record: BranchRecord) -> Node:
    return batch_node(
        name=record.branch_name, module="M4", consumes=DesignBatch, produces=QCReport,
        script=record.script, build_rows=_build_rows,
        parse=_common.make_parse(record.branch_name),
        description=record.description, venv_path=record.venv_path)


ADAPTER = Adapter(name="m4-qc-energy", consumes=DesignBatch, produces=QCReport, make_node=make_node)


def record(venv_path: str | None = None) -> BranchRecord:
    """The PyRosetta interface-ΔG branch (not a baseline; promoted alternative)."""
    return BranchRecord(
        module="M4", step="qc", branch_name="pyrosetta-dG", adapter="m4-qc-energy",
        description="Chai-1 co-fold + PyRosetta interface ΔG/SC (ref2015, repacked)",
        script=SCRIPT_PATH.read_text(), venv_path=venv_path)
