"""Demo: a real M4-QC FORK with two orthogonal branches, run offline (mock).

    branch 1: chai-ipsae    — Chai-1 holo co-fold + ipSAE interface confidence
    branch 2: pyrosetta-dG  — Chai-1 co-fold + PyRosetta interface ΔG (physics)

Both are committed to the registry on the same `qc` step, so the SOP assembles a
fork the operator selects at. Run:  python examples/demo_m4_fork.py

NOTE the soundness caveat documented in asop/adapters/m4_qc_energy.py: ranking two
*evaluator* branches by their own survivor fraction selects for the more permissive
gate, not the better metric. This demo exercises the fork machinery; binding a
ground signal into the objective is the open item before trusting the choice.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ASOP_QC_MOCK", "1")

from asop.adapters import m4_qc, m4_qc_energy
from asop.batch import LocalBatchRunner
from asop.dag import Executor, RunContext
from asop.operator import BanditOperator
from asop.registry import Registry
from asop.schemas import Design, DesignBatch, TargetSpec
from asop.store import Store

logging.basicConfig(level=logging.WARNING, format="%(message)s")

work = Path(tempfile.mkdtemp(prefix="asop-m4fork-"))
store = Store(work / "campaign.db")
runner = LocalBatchRunner(workdir=work / "batches", env={"ASOP_QC_MOCK": "1"})
registry = Registry(work / "dag_registry.jsonl",
                    adapters={"m4-qc": m4_qc.ADAPTER, "m4-qc-energy": m4_qc_energy.ADAPTER})

# commit both QC branches on the same step -> a fork
registry.register(m4_qc.baseline_record())      # chai-ipsae (baseline)
registry.register(m4_qc_energy.record())        # pyrosetta-dG (alternative)
sop = registry.build_sop("v2", ["qc"])
fork = sop.steps[0]
print("qc fork branches:", sorted(fork.branches), "| default:", fork.default_branch)

target = TargetSpec(name="ALKBH1",
                    sequence="MKQLEDKVEELLSKNYHLENEVARLKKLVGERADALEEENKELKEQV",
                    hotspots=[101, 142, 180], cofactors=["FE2", "AKG"])
op = BanditOperator(epsilon=0.3, seed=3, history_path=work / "operator.json")

print("\noperator selecting across the M4 fork:")
for r in range(6):
    batch = DesignBatch(target=target, designs=[
        Design(design_id=f"r{r}-d{i}", sequence="GSAAEELKKRLEELAKK" + "A" * i, method="rfdiffusion")
        for i in range(8)])
    report, trace = Executor(select=op.select).run(
        sop, batch, RunContext(round=r, extra={"store": store, "runner": runner}))
    op.record("qc", trace[0].branch, report.objective)
    print(f"  round {r}: {trace[0].branch:<13} objective={report.objective:.3f}  "
          f"({report.objective_definition})")

print("\nbranch means:", {b: round(s.mean, 3) for b, s in op._stats['qc'].items()})
print("ipSAE rows:", len(store.metrics_for("ALKBH1", "ipsae")),
      "| ΔG rows:", len(store.metrics_for("ALKBH1", "dg_separated")))
