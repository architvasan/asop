"""End-to-end, offline demo of the asop control layer — no HPC, no LLM.

Run:  python -m examples.demo_sop      (from the repo root)
  or: python examples/demo_sop.py

It shows, on stub M3/M4 node bodies:
  1. a typed, linear SOP (M3 design -> M4 QC) executing;
  2. a process-engineer proposing a new M3 branch and the stretch-test gate
     accepting it only because it is non-regressing, yielding a NEW SOP version;
  3. the operator (ε-greedy bandit) choosing between the baseline and promoted
     branch across rounds and learning which wins;
  4. the resource arbiter refusing a PE reservation that would starve production;
  5. the typed module bus carrying an M7->M2 backward insight (and refusing an
     undeclared edge).

Everything is deterministic (seeded), so the printed objectives are stable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asop.bus import Bus
from asop.dag import Executor, Node, RunContext, SOP
from asop.engineer import ProbeSet, ProcessEngineer
from asop.modules import ResourceManager
from asop.operator import BanditOperator
from asop.schemas import (
    BranchProposal,
    Design,
    DesignBatch,
    MechInsight,
    Metric,
    QCReport,
    TargetSpec,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("asop.dag").setLevel(logging.WARNING)   # hush per-step traces

# --------------------------------------------------------------------------- #
# Stub module bodies (in a real campaign these call loamlite skills on HPC).
# A "method" tag drives a deterministic quality score so branches differ.
# --------------------------------------------------------------------------- #

# baseline sits below the 0.6 ipSAE gate; the proposed branch clears it, so the
# stretch-test sees a real non-regression win and the operator has something to learn.
METHOD_QUALITY = {"rfdiffusion": 0.55, "plm-diversify": 0.81}


def m3_design(method: str, n: int = 8):
    def body(ctx: RunContext, target: TargetSpec) -> DesignBatch:
        designs = [
            Design(design_id=f"{target.name}-{method}-{i}",
                   sequence="GS" + "A" * (10 + i), method=method)
            for i in range(n)
        ]
        return DesignBatch(target=target, designs=designs)
    return body


def m4_qc(ctx: RunContext, batch: DesignBatch) -> QCReport:
    metrics, survivors = [], []
    for d in batch.designs:
        q = METHOD_QUALITY.get(d.method, 0.5)
        passed = q >= 0.6
        metrics.append(Metric(design_id=d.design_id, name="ipSAE_proxy", value=q, gate_pass=passed))
        if passed:
            survivors.append(d.design_id)
    objective = len(survivors) / max(batch.size, 1)   # survivor fraction, higher=better
    return QCReport(target_name=batch.target.name, metrics=metrics, survivors=survivors,
                    objective=objective, objective_definition="fraction of designs passing ipSAE gate")


# --------------------------------------------------------------------------- #
# Build the baseline SOP: M3 (design) -> M4 (QC)
# --------------------------------------------------------------------------- #

design_baseline = Node(name="design", consumes=TargetSpec, produces=DesignBatch,
                       fn=m3_design("rfdiffusion"), module="M3")
qc = Node(name="qc", consumes=DesignBatch, produces=QCReport, fn=m4_qc, module="M4")
sop_v0 = SOP(version="v0", steps=(design_baseline, qc))

target = TargetSpec(name="ALKBH1", sequence="MAAA...", hotspots=[101, 142, 180],
                    cofactors=["FE2", "AKG"])

print("\n=== 1. baseline SOP v0 ===")
final, trace = Executor().run(sop_v0, target)
print(f"objective={final.objective:.3f}  trace={[(t.step, t.branch) for t in trace]}")


# --------------------------------------------------------------------------- #
# 2. Process-engineer proposes a PLM-diversify M3 branch; stretch-test gates it.
# --------------------------------------------------------------------------- #

print("\n=== 2. process-engineer proposes M3.plm-diversify ===")
candidate = Node(name="plm-diversify", consumes=TargetSpec, produces=DesignBatch,
                 fn=m3_design("plm-diversify"), module="M3")
proposal = BranchProposal(module="M3", step_name="design", branch_name="plm-diversify",
                          description="ESM-2 masked-LM seed diversification of RFdiffusion backbones")
probes = ProbeSet(inputs=[
    TargetSpec(name="probe-KDM", sequence="M...", hotspots=[55]),
    TargetSpec(name="probe-TET", sequence="M...", hotspots=[88]),
])
eng = ProcessEngineer(module="M3", node_budget=2)
result, sop_v1 = eng.evaluate(sop_v0, proposal, candidate, probes, RunContext())
print(f"accepted={result.accepted}  baseline={result.baseline_objective:.3f}  "
      f"candidate={result.candidate_objective:.3f}")
print(f"reason: {result.reason}")
assert result.accepted and sop_v1 is not None
print(f"new SOP version: {sop_v1.version}  (fork 'design' now open to operator)")


# --------------------------------------------------------------------------- #
# 3. Operator learns which branch wins across rounds on the promoted SOP.
# --------------------------------------------------------------------------- #

print("\n=== 3. operator (ε-greedy) over rounds on SOP v1 ===")
op = BanditOperator(epsilon=0.3, seed=7)
fork = sop_v1.steps[0]        # the widened 'design' fork
for r in range(6):
    ctx = RunContext(round=r, dag_version=sop_v1.version)
    final, trace = Executor(select=op.select).run(sop_v1, target, ctx)
    chosen = trace[0]
    op.record(fork.name, chosen.branch, final.objective)
    print(f"round {r}: chose {chosen.branch:<13} objective={final.objective:.3f}  ({chosen.rationale})")
print("learned branch means:",
      {b: round(s.mean, 3) for b, s in op._stats[fork.name].items()})


# --------------------------------------------------------------------------- #
# 4. Resource arbiter protects production from PE starvation.
# --------------------------------------------------------------------------- #

print("\n=== 4. resource arbiter ===")
rm = ResourceManager(total_nodes=8, production_floor=2)
print("production reserves 4:", rm.reserve("production", 4, production=True))  # 4 free left
print("PE asks for 2 (ok):   ", rm.reserve("pe-M3", 2))                        # 2 free, >= floor 2
print("PE asks for 3 (deny): ", rm.reserve("pe-M5", 3), "-> would breach floor")  # would leave -1
print("snapshot:", rm.snapshot())


# --------------------------------------------------------------------------- #
# 5. Typed module bus: M7->M2 backward insight; undeclared edge refused.
# --------------------------------------------------------------------------- #

print("\n=== 5. module bus ===")
bus = Bus()
insight = MechInsight(target_name="ALKBH1",
                      statement="residence time (k_off) discriminates leads; ipTM does not",
                      changes_scoring=True)
bus.send("M7", "M2", "backward", topic="rescore-by-kinetics", payload=insight)
print("M2 backward inbox:", [(m.src, m.topic) for m in bus.inbox("M2", direction="backward")])
try:
    bus.send("M4", "M7", "forward", topic="nope")
except PermissionError as exc:
    print("refused undeclared edge:", exc)

print("\nOK")
