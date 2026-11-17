"""M4-QC branch 1: Chai-1 holo co-fold + ipSAE/ipTM interface confidence.

Wires the self-contained worker `asop/scripts/m4_qc_chai.py` into the DAG. The
store-query and result-parsing are shared with every other QC branch (`_common`);
this module owns only the worker script and the holo-cofactor mapping.
"""

from __future__ import annotations

from pathlib import Path

from ..dag import Node
from ..nodes import batch_node
from ..registry import Adapter, BranchRecord
from ..schemas import DesignBatch, QCReport
from ..store import Store
from . import _common

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "m4_qc_chai.py"

# Holo cofactors for metalloenzyme active sites, as Chai ligand SMILES. ALKBH1 is
# an Fe(II)/2-oxoglutarate dioxygenase, so fold it with Fe2+ + 2OG in the pocket.
COFACTOR_SMILES: dict[str, dict[str, str]] = {
    "FE2": {"name": "FE2", "smiles": "[Fe+2]"},
    "FE":  {"name": "FE",  "smiles": "[Fe+2]"},
    "ZN":  {"name": "ZN",  "smiles": "[Zn+2]"},
    "MG":  {"name": "MG",  "smiles": "[Mg+2]"},
    "MN":  {"name": "MN",  "smiles": "[Mn+2]"},
    "AKG": {"name": "AKG", "smiles": "O=C(O)CCC(=O)C(=O)O"},   # 2-oxoglutarate / α-KG
    "2OG": {"name": "2OG", "smiles": "O=C(O)CCC(=O)C(=O)O"},
}


def cofactor_ligands(cofactors: list[str]) -> list[dict[str, str]]:
    """Map target cofactor codes to Chai ligand entities; unknown codes are skipped
    (a missing cofactor is better than a wrong SMILES silently folded in)."""
    out = []
    for c in cofactors:
        lig = COFACTOR_SMILES.get(c.upper())
        if lig is not None:
            out.append(dict(lig))
    return out


def _build_rows(store: Store, inp: DesignBatch):
    return _common.make_build_rows()(store, inp, cofactor_ligands(inp.target.cofactors))


def make_node(record: BranchRecord) -> Node:
    """Materialize a registered M4-QC/ipSAE branch. The scientific body is
    `record.script` (frozen in the registry); the glue is shared `_common`."""
    return batch_node(
        name=record.branch_name, module="M4", consumes=DesignBatch, produces=QCReport,
        script=record.script, build_rows=_build_rows,
        parse=_common.make_parse(record.branch_name),
        description=record.description, venv_path=record.venv_path)


ADAPTER = Adapter(name="m4-qc", consumes=DesignBatch, produces=QCReport, make_node=make_node)


def baseline_record(venv_path: str | None = None) -> BranchRecord:
    """The default M4-QC branch: Chai-1 holo co-fold + ipSAE."""
    return BranchRecord(
        module="M4", step="qc", branch_name="chai-ipsae", adapter="m4-qc", is_baseline=True,
        description="Chai-1 holo co-fold + ipSAE/ipTM interface score",
        script=SCRIPT_PATH.read_text(), venv_path=venv_path)
