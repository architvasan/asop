"""Capstone demo: the operator / process-engineer / registry triangle.

    1. A baseline M4-QC branch (real script) is committed to the registry.
    2. The production SOP is ASSEMBLED FROM THE REGISTRY; the operator runs it.
    3. A process-engineer WRITES A NEW REAL SCRIPT, the stretch gate proves it
       non-regressing, and it is COMMITTED TO THE REGISTRY.
    4. The SOP is re-assembled from the registry — now a 2-branch fork — and the
       operator (choice-only interface) learns which registered branch wins.

Run:  python examples/demo_registry.py

The engineer uses skills/creativity to author the script; the operator never
sees code, only branch names + descriptions + objective history. Validated code
becomes part of the DAG only by passing the gate and being written to the
registry — exactly the contract requested.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asop.batch import BatchReceipt, LocalBatchRunner
from asop.dag import Executor, RunContext
from asop.engineer import ProbeSet, ProcessEngineer
from asop.nodes import batch_node
from asop.operator import BanditOperator
from asop.registry import Adapter, BranchRecord, Registry
from asop.schemas import BranchProposal, Design, DesignBatch, Metric, QCReport, TargetSpec
from asop.store import Store

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("asop.dag").setLevel(logging.WARNING)

# ----------------------------------------------------------------------------- #
# The M4-QC adapter: module-standard glue that turns ANY qc script into a node.
# An engineer only writes the `script`; build_rows/parse are fixed infrastructure.
# ----------------------------------------------------------------------------- #

# run_batch per-row contract: python step.py input.json -> result.json in cwd
QC_SCRIPT = '''
import json, sys
BONUS = {bonus}
row = json.load(open(sys.argv[1]))
ipsae = round(0.50 + (len(row["sequence"]) % 5) / 10.0 + BONUS, 3)
json.dump({{"design_id": row["design_id"], "ipsae": ipsae,
           "gate_pass": int(ipsae >= 0.60)}}, open("result.json", "w"))
'''


def _build_rows(store: Store, inp: DesignBatch):
    # deterministic: read work items straight off the typed artifact, and mirror
    # them into the store for provenance. (The batch_node demo shows the
    # read-from-DB-by-status variant; either is fine, both are LLM-free.)
    store.put_designs([{"design_id": d.design_id, "target": inp.target.name,
                        "sequence": d.sequence, "method": d.method} for d in inp.designs])
    return [{"design_id": d.design_id, "sequence": d.sequence} for d in inp.designs]


def _make_parse(run_id: str):
    def parse(receipt: BatchReceipt, inp: DesignBatch, store: Store) -> QCReport:
        metrics, survivors = [], []
        for r in receipt.rows:
            store.put_metrics([{"design_id": r["design_id"], "name": "ipsae",
                                "value": r["ipsae"], "gate_pass": r["gate_pass"], "run_id": run_id}])
            metrics.append(Metric(design_id=r["design_id"], name="ipsae",
                                  value=r["ipsae"], gate_pass=bool(r["gate_pass"])))
            if r["gate_pass"]:
                survivors.append(r["design_id"])
        obj = len(survivors) / max(len(receipt.rows), 1)
        return QCReport(target_name=inp.target.name, metrics=metrics, survivors=survivors,
                        objective=obj, objective_definition="fraction passing ipSAE>=0.60")
    return parse


def qc_make_node(record: BranchRecord):
    return batch_node(name=record.branch_name, module=record.module,
                      consumes=DesignBatch, produces=QCReport, script=record.script,
                      build_rows=_build_rows, parse=_make_parse(record.branch_name),
                      description=record.description, venv_path=record.venv_path)


QC_ADAPTER = Adapter(name="m4-qc", consumes=DesignBatch, produces=QCReport, make_node=qc_make_node)

# ----------------------------------------------------------------------------- #
# Wiring
# ----------------------------------------------------------------------------- #

work = Path(tempfile.mkdtemp(prefix="asop-reg-"))
store = Store(work / "campaign.db")
runner = LocalBatchRunner(workdir=work / "batches")
registry = Registry(work / "dag_registry.jsonl", adapters={"m4-qc": QC_ADAPTER})
op = BanditOperator(epsilon=0.3, seed=2, history_path=work / "operator.json")
STEPS = ["qc"]


def batch(target: TargetSpec, r: int) -> DesignBatch:
    return DesignBatch(target=target, designs=[
        Design(design_id=f"{target.name}-r{r}-d{i}", sequence="GS" + "A" * (10 + i), method="rfdiffusion")
        for i in range(6)])


def ctx(r: int, st: Store) -> RunContext:
    return RunContext(round=r, extra={"store": st, "runner": runner})


# 1. commit the baseline branch (real code) to the registry
registry.register(BranchRecord(module="M4", step="qc", branch_name="chai-fast", adapter="m4-qc",
                               description="Chai-1 ipSAE, 1 seed (cheap)", is_baseline=True,
                               script=QC_SCRIPT.format(bonus=0.00)))

print("\n=== 2. production SOP assembled from the registry (baseline only) ===")
target = TargetSpec(name="ALKBH1", sequence="MAAA", hotspots=[101])
sop = registry.build_sop("v1", STEPS)
for r in range(2):
    report, trace = Executor(select=op.select).run(sop, batch(target, r), ctx(r, store))
    op.record("qc", trace[0].branch, report.objective)
    print(f"round {r}: ran {trace[0].branch:<11} objective={report.objective:.3f}")

# 3. a process-engineer writes a NEW real script and submits it through the gate
print("\n=== 3. process-engineer proposes M4.boltz-xtra (writes real code) ===")
new_script = QC_SCRIPT.format(bonus=0.15)
candidate_rec = BranchRecord(module="M4", step="qc", branch_name="boltz-xtra", adapter="m4-qc",
                             description="Boltz + extra recycles (pricier, higher recall)",
                             script=new_script)
candidate_node = registry.materialize(candidate_rec)      # materialize WITHOUT registering yet
proposal = BranchProposal(module="M4", step_name="qc", branch_name="boltz-xtra",
                          description=candidate_rec.description)
probes = ProbeSet(inputs=[batch(TargetSpec(name="probe-KDM", sequence="M"), 0),
                          batch(TargetSpec(name="probe-TET", sequence="M"), 0)])
val_store = Store(work / "validation.db")                  # isolate validation side effects
eng = ProcessEngineer(module="M4")
result, _cand_sop = eng.evaluate(sop, proposal, candidate_node, probes, ctx(0, val_store))
print(f"baseline={result.baseline_objective:.3f}  candidate={result.candidate_objective:.3f}  "
      f"accepted={result.accepted}")

if result.accepted:
    registry.commit_promotion(result, adapter="m4-qc", script=new_script,
                              description=candidate_rec.description, proposed_by="pe-M4-evaluation")
    print(f"committed boltz-xtra to the registry at {result.new_dag_version}")

# 4. re-assemble the SOP from the registry — now a 2-branch fork — and run it
print("\n=== 4. SOP re-assembled from registry (fork); operator selects ===")
sop = registry.build_sop(result.new_dag_version or "v2", STEPS)
for r in range(2, 8):
    report, trace = Executor(select=op.select).run(sop, batch(target, r), ctx(r, store))
    op.record("qc", trace[0].branch, report.objective)
    print(f"round {r}: ran {trace[0].branch:<11} objective={report.objective:.3f}  ({trace[0].rationale})")

print("\nlearned branch means:", {b: round(s.mean, 3) for b, s in op._stats['qc'].items()})
print("registry now holds:", [r.branch_name for r in registry.branches('qc')])
print("registry file:", registry.path)
