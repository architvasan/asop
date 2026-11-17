"""GraphSOP: the non-linear scale-out path for the SOP, parallel to `dag.SOP`.

`dag.SOP` is a *linear* typed chain (a tuple of steps). It is the right shape for
the M1→M4 production loop and stays the default. This module is the representation
to reach for when the pipeline stops being a line — real fan-out, joins, and
diamonds (e.g. design → {fold, developability, dock} → merge → rank).

Design, following the scaling principle "structure goes in the graph, data
parallelism goes in the node":

- A `GraphNode` has **named, typed input ports** (not one `consumes`) and one
  output type. Its body is `fn(ctx, {port: artifact}) -> artifact`. Per-item
  volume (thousands of designs) still fans out *inside* a node via `run_batch`,
  never as graph structure — so graph size stays O(methods), execution O(items).
- Edges wire one node's single output into a named input port of another.
- A `GraphFork` is a choice point: interchangeable branches sharing the fork's
  input ports and output type. It reuses the *exact* `Selector` contract and the
  operators from `operator.py` unchanged (they only read `.name`, `.branches`,
  and each branch's `.description`).
- `GraphSOP` validates on construction: every edge is type-correct, every input
  port is fed exactly once (by an edge or left open for a run-time seed), and the
  graph is **acyclic** — checked for free by `graphlib.TopologicalSorter`.
- `GraphExecutor` runs nodes in topological order, threading each output to its
  consumers and recording the same `TraceEntry` provenance as `dag.Executor`.

Interop: `lift()` wraps an existing single-input `dag.Node`/`dag.Fork` as a
graph step, and `GraphSOP.from_linear()` turns any `dag.SOP` into an equivalent
GraphSOP — so the two paths produce identical results on a linear pipeline (see
`tests/test_graph.py`). Nothing in `dag.py` changes.
"""

from __future__ import annotations

import graphlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .dag import Fork, Node, RunContext, Selector, TraceEntry, _default_select
from .schemas import Artifact

log = logging.getLogger("asop.graph")

# A graph node body: (ctx, {port_name: artifact}) -> output artifact.
GraphFn = Callable[[RunContext, dict[str, Artifact]], Artifact]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphNode:
    """A node with named, typed input ports and a single typed output."""

    name: str
    inputs: Mapping[str, type[Artifact]]      # port name -> required artifact type
    produces: type[Artifact]
    fn: GraphFn
    module: str = ""
    description: str = ""

    def input_ports(self) -> tuple[str, ...]:
        return tuple(self.inputs)

    def run(self, ctx: RunContext, inbound: dict[str, Artifact]) -> Artifact:
        for port, typ in self.inputs.items():
            if port not in inbound:
                raise ValueError(f"node {self.name!r} missing input port {port!r}")
            if not isinstance(inbound[port], typ):
                raise TypeError(
                    f"node {self.name!r} port {port!r} wants {typ.__name__}, "
                    f"got {type(inbound[port]).__name__}"
                )
        out = self.fn(ctx, inbound)
        if not isinstance(out, self.produces):
            raise TypeError(
                f"node {self.name!r} must produce {self.produces.__name__}, "
                f"got {type(out).__name__}"
            )
        # Same provenance stamp as dag.Node.run.
        return out.model_copy(update={"origin": f"{ctx.dag_version}:{self.name}"})


@dataclass(frozen=True)
class GraphFork:
    """A choice point whose branches share the fork's input ports and output type,
    so they are interchangeable by construction (as `dag.Fork`). Attribute names
    (`name`, `branches`, `default_branch`, `allow_operator`) match `dag.Fork` so
    the existing operators and `_default_select` work on it unchanged."""

    name: str
    inputs: Mapping[str, type[Artifact]]
    produces: type[Artifact]
    branches: dict[str, GraphNode]
    allow_operator: bool = False
    default_branch: str | None = None
    module: str = ""

    def input_ports(self) -> tuple[str, ...]:
        return tuple(self.inputs)

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError(f"fork {self.name!r} has no branches")
        for bname, b in self.branches.items():
            if dict(b.inputs) != dict(self.inputs) or b.produces is not self.produces:
                raise TypeError(
                    f"branch {bname!r} of fork {self.name!r} has I/O "
                    f"{dict(b.inputs)}->{b.produces.__name__}, fork requires "
                    f"{dict(self.inputs)}->{self.produces.__name__}"
                )
        if self.default_branch is not None and self.default_branch not in self.branches:
            raise ValueError(
                f"fork {self.name!r} default_branch {self.default_branch!r} is not a branch"
            )


GraphStep = GraphNode | GraphFork


@dataclass(frozen=True)
class Edge:
    """Wire the single output of `src` into input port `port` of `dst`."""

    src: str
    dst: str
    port: str = "in"


# --------------------------------------------------------------------------- #
# GraphSOP
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphSOP:
    """An immutable, versioned, type-checked DAG of steps. Validated and
    topologically ordered on construction."""

    version: str
    nodes: Mapping[str, GraphStep]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a GraphSOP needs at least one node")

        # 1. edges reference real nodes and real ports; no port fed twice.
        fed: dict[tuple[str, str], str] = {}        # (dst, port) -> src
        for e in self.edges:
            if e.src not in self.nodes:
                raise KeyError(f"edge src {e.src!r} is not a node")
            if e.dst not in self.nodes:
                raise KeyError(f"edge dst {e.dst!r} is not a node")
            if e.port not in self.nodes[e.dst].input_ports():
                raise KeyError(
                    f"edge into {e.dst!r} names port {e.port!r}; "
                    f"node has ports {self.nodes[e.dst].input_ports()}"
                )
            key = (e.dst, e.port)
            if key in fed:
                raise ValueError(
                    f"port {e.port!r} of {e.dst!r} is fed twice "
                    f"(by {fed[key]!r} and {e.src!r})"
                )
            fed[key] = e.src

        # 2. edge type compatibility: producer.produces must match consumer port.
        for e in self.edges:
            want = self.nodes[e.dst].inputs[e.port]
            got = self.nodes[e.src].produces
            if got is not want:
                raise TypeError(
                    f"edge {e.src}->{e.dst}.{e.port}: {e.src} produces "
                    f"{got.__name__} but port wants {want.__name__}"
                )

        # 3. acyclicity + a topological order, for free, from graphlib.
        deps: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            deps[e.dst].add(e.src)
        try:
            order = tuple(graphlib.TopologicalSorter(deps).static_order())
        except graphlib.CycleError as exc:
            raise ValueError(f"GraphSOP {self.version} has a cycle: {exc.args[1]}") from exc
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_fed", fed)

    # -- introspection ------------------------------------------------------- #

    @property
    def order(self) -> tuple[str, ...]:
        """A valid execution order (topological)."""
        return self._order  # type: ignore[attr-defined]

    def open_ports(self) -> list[tuple[str, str]]:
        """Input ports with no inbound edge — the graph's entry points, which must
        be supplied as seeds at run time."""
        fed = self._fed  # type: ignore[attr-defined]
        return [(n, p) for n, step in self.nodes.items()
                for p in step.input_ports() if (n, p) not in fed]

    def sinks(self) -> list[str]:
        """Nodes whose output no edge consumes — the graph's results."""
        consumed = {e.src for e in self.edges}
        return [n for n in self.nodes if n not in consumed]

    def edge_into(self, node: str, port: str) -> str | None:
        return self._fed.get((node, port))  # type: ignore[attr-defined]

    # -- interop ------------------------------------------------------------- #

    @classmethod
    def from_linear(cls, sop, *, port: str = "in") -> "GraphSOP":
        """Build an equivalent GraphSOP from a linear `dag.SOP` by lifting each
        step and chaining consecutive steps with a single edge."""
        nodes: dict[str, GraphStep] = {s.name: lift(s, port=port) for s in sop.steps}
        edges = tuple(Edge(src=sop.steps[i - 1].name, dst=sop.steps[i].name, port=port)
                      for i in range(1, len(sop.steps)))
        return cls(version=sop.version, nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- #
# Lifting linear steps into the graph world
# --------------------------------------------------------------------------- #


def lift(step: Node | Fork, *, port: str = "in") -> GraphStep:
    """Wrap a single-input `dag.Node`/`dag.Fork` as a one-port graph step. The
    wrapped body calls the original `fn` (NOT `.run`) so provenance is stamped once
    by the graph layer, matching `dag`'s single stamp."""
    if isinstance(step, Fork):
        branches = {name: _lift_node(b, port=port) for name, b in step.branches.items()}
        return GraphFork(
            name=step.name, inputs={port: step.consumes}, produces=step.produces,
            branches=branches, allow_operator=step.allow_operator,
            default_branch=step.default_branch, module=step.module,
        )
    return _lift_node(step, port=port)


def _lift_node(node: Node, *, port: str) -> GraphNode:
    fn = node.fn
    return GraphNode(
        name=node.name, inputs={port: node.consumes}, produces=node.produces,
        fn=lambda ctx, inbound, _fn=fn, _p=port: _fn(ctx, inbound[_p]),
        module=node.module, description=node.description,
    )


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class GraphExecutor:
    """Runs a GraphSOP in topological order. The only place an agent influences
    control flow is `select` at an allowed `GraphFork`; everything else is
    deterministic. Mirrors `dag.Executor`."""

    def __init__(self, select: Selector | None = None):
        self.select = select or _default_select

    def run(self, graph: GraphSOP, seeds: dict[str, Artifact],
            ctx: RunContext | None = None
            ) -> tuple[dict[str, Artifact], list[TraceEntry]]:
        """`seeds` supplies the open input ports, keyed ``"node.port"``. Returns
        ``(outputs, trace)`` where `outputs` maps each sink node to its artifact."""
        ctx = ctx or RunContext(dag_version=graph.version)
        import dataclasses
        ctx = dataclasses.replace(ctx, dag_version=graph.version)

        # validate seeds cover exactly the open ports, with correct types.
        open_ports = {f"{n}.{p}": graph.nodes[n].inputs[p] for n, p in graph.open_ports()}
        missing = set(open_ports) - set(seeds)
        if missing:
            raise ValueError(f"GraphSOP {graph.version} missing seeds for {sorted(missing)}")
        for key, typ in open_ports.items():
            if not isinstance(seeds[key], typ):
                raise TypeError(f"seed {key!r} must be {typ.__name__}, "
                                f"got {type(seeds[key]).__name__}")

        results: dict[str, Artifact] = {}
        trace: list[TraceEntry] = []
        for nid in graph.order:
            step = graph.nodes[nid]
            inbound: dict[str, Artifact] = {}
            for p in step.input_ports():
                src = graph.edge_into(nid, p)
                inbound[p] = results[src] if src is not None else seeds[f"{nid}.{p}"]

            if isinstance(step, GraphFork):
                bname, chosen_by, why = self._choose(ctx, step)
                out = step.branches[bname].run(ctx, inbound)
                trace.append(TraceEntry(nid, bname, chosen_by, why, type(out).__name__))
            else:
                out = step.run(ctx, inbound)
                trace.append(TraceEntry(nid, None, "node", "", type(out).__name__))
            results[nid] = out
            ctx.logger.info("GraphSOP %s: %s -> %s", graph.version, nid, type(out).__name__)

        return {n: results[n] for n in graph.sinks()}, trace

    def _choose(self, ctx: RunContext, fork: GraphFork) -> tuple[str, str, str]:
        if not fork.allow_operator:
            return _default_select(ctx, fork)        # closed fork: deterministic
        name, chosen_by, why = self.select(ctx, fork)
        if name not in fork.branches:
            raise ValueError(
                f"selector picked branch {name!r} not in fork {fork.name!r} "
                f"(have {sorted(fork.branches)})"
            )
        return name, chosen_by, why
