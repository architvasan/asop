"""Demo: the REAL M4-QC branch (Chai-1 co-fold + ipSAE) assembled from the
registry and run through the DAG — in mock mode so it runs offline.

Run:  ASOP_QC_MOCK=1 python examples/demo_m4_qc.py
  (the script sets the env itself if you forget)

Shows the production shape of item 1: the baseline M4-QC branch is committed to
the registry, the SOP is assembled from it, and the operator runs it over an
ALKBH1 design batch folded HOLO (Fe2+ + 2OG in the pocket). Swap
`LocalBatchRunner(env=mock)` for `McpBatchRunner(mcp_url, venv_path)` and drop the
mock env to run the identical pipeline on Aurora.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ASOP_QC_MOCK", "1")

from asop.adapters import m4_qc
from asop.batch import LocalBatchRunner
from asop.dag import Executor, RunContext
from asop.registry import Registry
from asop.schemas import Design, DesignBatch, TargetSpec
from asop.store import Store

logging.basicConfig(level=logging.WARNING, format="%(message)s")

work = Path(tempfile.mkdtemp(prefix="asop-m4-"))
store = Store(work / "campaign.db")
runner = LocalBatchRunner(workdir=work / "batches", env={"ASOP_QC_MOCK": "1"})
registry = Registry(work / "dag_registry.jsonl", adapters={"m4-qc": m4_qc.ADAPTER})

# 1. commit the real baseline M4-QC branch (the frozen worker script)
registry.register(m4_qc.baseline_record())
print("registered:", [r.branch_name for r in registry.branches("qc")])

# 2. assemble the production SOP from the registry
sop = registry.build_sop("v1", ["qc"])

# 3. an ALKBH1 design batch, folded HOLO (Fe2+ + 2-oxoglutarate)
target = TargetSpec(name="ALKBH1",
                    sequence="MKQLEDKVEELLSKNYHLENEVARLKKLVGERADALEEENKELKEQV",
                    hotspots=[101, 142, 180], cofactors=["FE2", "AKG"])
batch = DesignBatch(target=target, designs=[
    Design(design_id=f"alkbh1-d{i}", sequence="GSAAEELKKRLEELAKK" + "A" * i, method="rfdiffusion")
    for i in range(8)])

# show the holo cofactors attached to every QC row
print("holo ligands:", m4_qc.cofactor_ligands(target.cofactors))

# 4. run the DAG
report, trace = Executor().run(sop, batch, RunContext(extra={"store": store, "runner": runner}))
print(f"\nobjective={report.objective:.3f}  ({report.objective_definition})")
print(f"survivors: {len(report.survivors)}/{batch.size}")
for m in [m for m in report.metrics if m.name == "ipsae"][:4]:
    print(f"  {m.design_id}: ipSAE={m.value}  gate_pass={m.gate_pass}")
print("ipSAE metric rows in store:", len(store.metrics_for("ALKBH1", "ipsae")))
