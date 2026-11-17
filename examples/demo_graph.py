"""Demo: the non-linear GraphSOP scale-out path (asop/graph.py), run offline.

A diamond that the linear dag.SOP cannot express:

        design ──► fold ─────────────┐
               └──► developability ──┴──► rank   (fork: by-confidence | by-balance)

- fan-out: `design` feeds two parallel evaluators;
- join:    `rank` has two typed input ports and merges them;
- fork:    `rank` is a choice point the operator selects at.

Run:  python examples/demo_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asop.graph import Edge, GraphExecutor, GraphFork, GraphNode, GraphSOP
from asop.operator import BanditOperator
from asop.schemas import Artifact
from asop.dag import RunContext


class DesignBatch(Artifact):
    n: int = 0
class FoldScore(Artifact):
    conf: float = 0.0
class DevScore(Artifact):
    dev: float = 0.0
class Ranking(Artifact):
    score: float = 0.0
    policy: str = ""


design = GraphNode("design", {"in": DesignBatch}, DesignBatch,
                   lambda c, d: DesignBatch(n=d["in"].n), description="pass-through seed")
fold = GraphNode("fold", {"in": DesignBatch}, FoldScore,
                 lambda c, d: FoldScore(conf=0.5 + 0.1 * d["in"].n), description="chai ipSAE")
dev = GraphNode("developability", {"in": DesignBatch}, DevScore,
                lambda c, d: DevScore(dev=1.0 - 0.05 * d["in"].n), description="liability scan")
rank = GraphFork(
    "rank", {"f": FoldScore, "d": DevScore}, Ranking,
    branches={
        "by-confidence": GraphNode("rank", {"f": FoldScore, "d": DevScore}, Ranking,
                                   lambda c, x: Ranking(score=x["f"].conf, policy="confidence"),
                                   description="rank by fold confidence only"),
        "by-balance": GraphNode("rank", {"f": FoldScore, "d": DevScore}, Ranking,
                                lambda c, x: Ranking(score=0.5 * (x["f"].conf + x["d"].dev),
                                                     policy="balance"),
                                description="balance confidence and developability"),
    },
    allow_operator=True, default_branch="by-confidence")

g = GraphSOP(
    version="diamond-v1",
    nodes={"design": design, "fold": fold, "developability": dev, "rank": rank},
    edges=(Edge("design", "fold", "in"), Edge("design", "developability", "in"),
           Edge("fold", "rank", "f"), Edge("developability", "rank", "d")))

print("topological order :", list(g.order))
print("open input ports  :", g.open_ports())
print("sinks (results)   :", g.sinks())

op = BanditOperator(epsilon=0.3, seed=1)
print("\noperator selecting at the join-fork across rounds:")
for r in range(6):
    out, trace = GraphExecutor(select=op.select).run(
        g, {"design.in": DesignBatch(n=3)}, RunContext(round=r, dag_version=g.version))
    ranking = out["rank"]
    op.record("rank", ranking.policy and next(t.branch for t in trace if t.step == "rank"),
              ranking.score)
    chosen = next(t.branch for t in trace if t.step == "rank")
    print(f"  round {r}: rank={chosen:<13} score={ranking.score:.3f} ({ranking.policy})")

print("\nbranch means:", {b: round(s.mean, 3) for b, s in op._stats['rank'].items()})
