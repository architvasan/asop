"""Shared M4-QC glue, so every QC branch builds its rows and folds its results the
same way. A QC branch differs only in its worker *script* (ipSAE vs PyRosetta vs
…); the store-query and result-parsing below are identical, which is also what
keeps the branches' reported objective a common currency.

Metric-agnostic by design: a worker emits whatever scalar metrics it computes,
plus a top-level ``gate_pass`` and ``objective_definition``. `parse` records every
recognized metric, counts survivors by ``gate_pass``, and sets the campaign
objective to the survivor fraction. Adding a new QC branch needs no change here.
"""

from __future__ import annotations

from typing import Any

from ..batch import BatchReceipt
from ..schemas import DesignBatch, Metric, QCReport
from ..store import Store

# Every scalar metric any QC worker may emit. Unknown keys in a result are ignored
# (id/flags/paths), so a new metric is added by naming it here.
METRIC_KEYS: frozenset[str] = frozenset({
    "ipsae", "ipsae_ab", "ipsae_ba", "iptm", "ptm", "if_plddt",       # ipSAE branch
    "dg_separated", "dsasa", "sc", "hbonds", "nres_interface",        # energy branch
})


def make_build_rows(seed: int = 42, num_samples: int = 5):
    """Rows are identical across QC branches; each worker reads its own gate with
    an internal default, so no branch-specific knobs leak in here. Cofactor
    ligands are attached by the caller's adapter (holo folding)."""

    def build_rows(store: Store, inp: DesignBatch, ligands: list[dict]) -> list[dict[str, Any]]:
        store.put_designs([{"design_id": d.design_id, "target": inp.target.name,
                            "sequence": d.sequence, "method": d.method} for d in inp.designs])
        return [{"design_id": d.design_id,
                 "target_sequence": inp.target.sequence,
                 "binder_sequence": d.sequence,
                 "ligands": ligands,
                 "seed": seed,
                 "num_samples": num_samples} for d in inp.designs]

    return build_rows


def make_parse(run_id: str):
    def parse(receipt: BatchReceipt, inp: DesignBatch, store: Store) -> QCReport:
        metrics: list[Metric] = []
        survivors: list[str] = []
        evaluated: list[str] = []
        obj_def = ""
        for r in receipt.rows:
            did = str(r.get("design_id", "unknown"))
            obj_def = obj_def or r.get("objective_definition", "")
            if r.get("error"):                      # a failed fold is a real 0, not a survivor
                continue
            evaluated.append(did)
            passed = bool(r.get("gate_pass"))
            for name in METRIC_KEYS & r.keys():
                store.put_metrics([{"design_id": did, "name": name, "value": float(r[name]),
                                    "gate_pass": int(passed), "run_id": run_id}])
                metrics.append(Metric(design_id=did, name=name, value=float(r[name]),
                                      gate_pass=passed))
            if passed:
                survivors.append(did)
        if evaluated:
            store.set_status(evaluated, "evaluated")
        n = len(receipt.rows)
        return QCReport(target_name=inp.target.name, metrics=metrics, survivors=survivors,
                        objective=(len(survivors) / n if n else 0.0),
                        objective_definition=obj_def or "fraction of designs passing the QC gate")

    return parse
