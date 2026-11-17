"""Demo: the operator-selects-real-functions pattern the campaign is built on.

No skills. No code generation. The operator only names a branch; the branch is a
frozen, deterministic function (`batch_node`) that:
  1. runs a deterministic query against the SQLite store ("designs awaiting QC"),
  2. fans them across HPC in ONE run_batch (here a LocalBatchRunner subprocess),
  3. writes metrics straight back to the store and returns a typed QCReport.

Run:  python examples/demo_batch_node.py

Both QC branches are real scripts; 'boltz-xtra' scores a bit higher, so the
operator should learn to prefer it — chosen purely from branch descriptions +
objective history, which is all the selection interface ever exposes.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asop.batch import BatchReceipt, LocalBatchRunner
from asop.dag import Executor, Fork, RunContext, SOP
from asop.nodes import batch_node
from asop.operator import BanditOperator
from asop.schemas import DesignBatch, Design, Metric, QCReport, TargetSpec
from asop.store import Store

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# --- a real batch script (same contract local or HPC: argv input.jsonl output.jsonl) --- #
# In production this would import a loamlite skill's runner (chai/boltz) and emit ipSAE.
# run_batch per-row contract: python step.py input.json -> result.json in cwd
QC_SCRIPT = '''
import json, sys
BONUS = {bonus}
row = json.load(open(sys.argv[1]))
ipsae = round(0.50 + (len(row["sequence"]) % 5) / 10.0 + BONUS, 3)   # deterministic stand-in
json.dump({{"design_id": row["design_id"], "ipsae": ipsae,
           "gate_pass": int(ipsae >= 0.60)}}, open("result.json", "w"))
'''


def build_rows(store: Store, inp: DesignBatch):
    # persist the batch's designs, then run the DETERMINISTIC work query
    store.put_designs([{"design_id": d.design_id, "target": inp.target.name,
                        "sequence": d.sequence, "method": d.method} for d in inp.designs])
    return [{"design_id": r["design_id"], "sequence": r["sequence"]}
            for r in store.designs(inp.target.name, status="new")]


def make_parse(run_id: str):
    def parse(receipt: BatchReceipt, inp: DesignBatch, store: Store) -> QCReport:
        metrics, survivors = [], []
        for r in receipt.rows:
            store.put_metrics([{"design_id": r["design_id"], "name": "ipsae",
                                "value": r["ipsae"], "gate_pass": r["gate_pass"], "run_id": run_id}])
            metrics.append(Metric(design_id=r["design_id"], name="ipsae",
                                  value=r["ipsae"], gate_pass=bool(r["gate_pass"])))
            if r["gate_pass"]:
                survivors.append(r["design_id"])
        store.set_status([r["design_id"] for r in receipt.rows], "evaluated")
        obj = len(survivors) / max(len(receipt.rows), 1)
        return QCReport(target_name=inp.target.name, metrics=metrics, survivors=survivors,
                        objective=obj, objective_definition="fraction passing ipSAE>=0.60")
    return parse


# --- two REAL, frozen QC branches the operator chooses between ----------------------- #
qc_fast = batch_node(name="chai-fast", module="M4", consumes=DesignBatch, produces=QCReport,
                     script=QC_SCRIPT.format(bonus=0.00), build_rows=build_rows,
                     parse=make_parse("chai-fast"), description="Chai-1 ipSAE, 1 seed (cheap)")
qc_xtra = batch_node(name="boltz-xtra", module="M4", consumes=DesignBatch, produces=QCReport,
                     script=QC_SCRIPT.format(bonus=0.15), build_rows=build_rows,
                     parse=make_parse("boltz-xtra"),
                     description="Boltz + extra recycles (pricier, higher recall)")

qc_fork = Fork(name="qc", consumes=DesignBatch, produces=QCReport,
               branches={"chai-fast": qc_fast, "boltz-xtra": qc_xtra},
               allow_operator=True, default_branch="chai-fast", module="M4")
sop = SOP(version="v1", steps=(qc_fork,))

# --- set up store + local runner, seed a design batch -------------------------------- #
work = Path(tempfile.mkdtemp(prefix="asop-demo-"))
store = Store(path=work / "campaign.db")
runner = LocalBatchRunner(workdir=work / "batches")
op = BanditOperator(epsilon=0.3, seed=1, history_path=work / "operator.json")

target = TargetSpec(name="ALKBH1", sequence="MAAA", hotspots=[101])

print("=== operator selects a REAL function; the function runs the batch ===")
for r in range(6):
    # each round M3 hands QC a fresh design batch (unique ids per round)
    batch = DesignBatch(target=target, designs=[
        Design(design_id=f"r{r}-d{i}", sequence="GS" + "A" * (10 + i), method="rfdiffusion")
        for i in range(6)])
    ctx = RunContext(round=r, dag_version=sop.version, extra={"store": store, "runner": runner})
    report, trace = Executor(select=op.select).run(sop, batch, ctx)
    op.record("qc", trace[0].branch, report.objective)
    print(f"round {r}: operator ran {trace[0].branch:<11} "
          f"objective={report.objective:.3f} survivors={len(report.survivors)}/6")

print("\nlearned branch means:",
      {b: round(s.mean, 3) for b, s in op._stats['qc'].items()})
print("metrics rows now in the store:", len(store.metrics_for("ALKBH1", "ipsae")))
print("batches recorded:", work / "campaign.db")
