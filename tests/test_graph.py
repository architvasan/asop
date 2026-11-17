"""Tests for the GraphSOP scale-out path (asop/graph.py).

Exercises a real non-linear DAG (fan-out + fork + join), the construction-time
invariants (type-checked edges, ports fed once, acyclicity), the reuse of the
existing operators at a GraphFork, and equivalence with the linear dag.SOP.
Runs fully offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from asop.dag import SOP, Executor, Node, RunContext
from asop.graph import Edge, GraphExecutor, GraphFork, GraphNode, GraphSOP, lift
from asop.operator import BanditOperator
from asop.schemas import Artifact


# -- tiny artifact types to keep the test about graph mechanics -------------- #
class Seed(Artifact):
    n: int = 0
class Mid(Artifact):
    n: int = 0
class Left(Artifact):
    n: int = 0
class Right(Artifact):
    n: int = 0
    tag: str = ""
class Out(Artifact):
    n: int = 0
    tag: str = ""


def _diamond(allow_operator: bool = True) -> GraphSOP:
    """seed → gen → {left, right(fork)} → merge.  (fan-out + fork + join)"""
    gen = GraphNode("gen", {"in": Seed}, Mid, lambda c, d: Mid(n=d["in"].n + 1))
    left = GraphNode("left", {"in": Mid}, Left, lambda c, d: Left(n=d["in"].n * 2))
    right = GraphFork(
        "right", {"in": Mid}, Right,
        branches={
            "rA": GraphNode("right", {"in": Mid}, Right,
                            lambda c, d: Right(n=d["in"].n + 10, tag="A"), description="branch A"),
            "rB": GraphNode("right", {"in": Mid}, Right,
                            lambda c, d: Right(n=d["in"].n + 20, tag="B"), description="branch B"),
        },
        allow_operator=allow_operator, default_branch="rA")
    merge = GraphNode("merge", {"l": Left, "r": Right}, Out,
                      lambda c, d: Out(n=d["l"].n + d["r"].n, tag=d["r"].tag))
    return GraphSOP(
        version="g1",
        nodes={"gen": gen, "left": left, "right": right, "merge": merge},
        edges=(Edge("gen", "left", "in"), Edge("gen", "right", "in"),
               Edge("left", "merge", "l"), Edge("right", "merge", "r")))


# --------------------------------------------------------------------------- #
# structure / introspection
# --------------------------------------------------------------------------- #


def test_topo_order_respects_dependencies():
    g = _diamond()
    pos = {n: i for i, n in enumerate(g.order)}
    assert pos["gen"] < pos["left"] < pos["merge"]
    assert pos["gen"] < pos["right"] < pos["merge"]


def test_open_ports_and_sinks():
    g = _diamond()
    assert g.open_ports() == [("gen", "in")]        # only entry point
    assert g.sinks() == ["merge"]                    # only result


def test_diamond_runs_with_fan_out_and_join():
    g = _diamond(allow_operator=False)               # closed fork -> default rA
    out, trace = GraphExecutor().run(g, {"gen.in": Seed(n=1)},
                                     RunContext(dag_version="g1"))
    # gen:2, left:4, rA:12  -> merge 16, tag A
    assert out["merge"].n == 16 and out["merge"].tag == "A"
    assert out["merge"].origin == "g1:merge"         # provenance stamped
    assert [t.step for t in trace] == list(g.order)


# --------------------------------------------------------------------------- #
# construction-time invariants
# --------------------------------------------------------------------------- #


def test_cycle_is_rejected():
    p = GraphNode("p", {"in": Mid}, Seed, lambda c, d: Seed())
    q = GraphNode("q", {"in": Seed}, Mid, lambda c, d: Mid())
    with pytest.raises(ValueError, match="cycle"):
        GraphSOP("c1", {"p": p, "q": q},
                 (Edge("q", "p", "in"), Edge("p", "q", "in")))


def test_edge_type_mismatch_is_rejected():
    gen = GraphNode("gen", {"in": Seed}, Mid, lambda c, d: Mid())
    sink = GraphNode("sink", {"in": Left}, Out, lambda c, d: Out())   # wants Left, fed Mid
    with pytest.raises(TypeError, match="wants Left"):
        GraphSOP("t1", {"gen": gen, "sink": sink}, (Edge("gen", "sink", "in"),))


def test_port_fed_twice_is_rejected():
    a = GraphNode("a", {"in": Seed}, Mid, lambda c, d: Mid())
    b = GraphNode("b", {"in": Seed}, Mid, lambda c, d: Mid())
    sink = GraphNode("sink", {"in": Mid}, Out, lambda c, d: Out())
    with pytest.raises(ValueError, match="fed twice"):
        GraphSOP("p1", {"a": a, "b": b, "sink": sink},
                 (Edge("a", "sink", "in"), Edge("b", "sink", "in")))


def test_unknown_port_is_rejected():
    gen = GraphNode("gen", {"in": Seed}, Mid, lambda c, d: Mid())
    sink = GraphNode("sink", {"in": Mid}, Out, lambda c, d: Out())
    with pytest.raises(KeyError, match="port"):
        GraphSOP("u1", {"gen": gen, "sink": sink}, (Edge("gen", "sink", "nope"),))


def test_missing_seed_is_rejected():
    g = _diamond(allow_operator=False)
    with pytest.raises(ValueError, match="missing seeds"):
        GraphExecutor().run(g, {}, RunContext(dag_version="g1"))


def test_bad_fork_branch_io_is_rejected():
    with pytest.raises(TypeError, match="branch"):
        GraphFork("f", {"in": Mid}, Right,
                  branches={"ok": GraphNode("f", {"in": Mid}, Right, lambda c, d: Right()),
                            "bad": GraphNode("f", {"in": Seed}, Right, lambda c, d: Right())})


# --------------------------------------------------------------------------- #
# operator reuse at a GraphFork
# --------------------------------------------------------------------------- #


def test_operator_selects_at_graph_fork():
    g = _diamond(allow_operator=True)
    op = BanditOperator(epsilon=0.3, seed=0)
    seen = set()
    for r in range(6):
        out, trace = GraphExecutor(select=op.select).run(
            g, {"gen.in": Seed(n=1)}, RunContext(round=r, dag_version="g1"))
        branch = next(t.branch for t in trace if t.step == "right")
        op.record("right", branch, float(out["merge"].n))     # feed objective back
        seen.add((branch, out["merge"].tag))
    # both branches were explored, and the chosen branch flows through the join
    assert {"rA", "rB"} == {b for b, _ in seen}
    assert ("rA", "A") in seen and ("rB", "B") in seen


# --------------------------------------------------------------------------- #
# equivalence with the linear dag.SOP
# --------------------------------------------------------------------------- #


def test_from_linear_matches_linear_executor():
    n1 = Node("n1", Seed, Mid, lambda c, a: Mid(n=a.n + 1))
    n2 = Node("n2", Mid, Out, lambda c, a: Out(n=a.n * 3, tag="z"))
    sop = SOP(version="lin1", steps=(n1, n2))

    lin_out, lin_trace = Executor().run(sop, Seed(n=4), RunContext(dag_version="lin1"))

    g = GraphSOP.from_linear(sop)
    assert g.open_ports() == [("n1", "in")] and g.sinks() == ["n2"]
    g_out, g_trace = GraphExecutor().run(g, {"n1.in": Seed(n=4)}, RunContext(dag_version="lin1"))

    assert g_out["n2"].model_dump() == lin_out.model_dump()      # identical result + origin
    assert [t.step for t in g_trace] == [t.step for t in lin_trace]
