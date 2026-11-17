"""The SOP-DAG: a versioned, immutable pipeline of typed steps.

Mirrors loamlite's central decision — *the orchestrator is plain Python* — one
level up. A `SOP` is a fixed, ordered sequence of `Step`s. A step is either a
`Node` (exactly one implementation) or a `Fork` (several interchangeable
branches, one picked at run time by the operator). There is nothing for an LLM
to decide about scheduling; the only run-time decision is *which branch* at an
allowed fork, and that decision is bounded to type-compatible alternatives.

Why typing is load-bearing (README critique point 7): every step declares the
artifact type it `consumes` and the type it `produces`. `SOP.__init__` checks
that consecutive steps chain, and `Fork` checks that every branch shares the
fork's `consumes`/`produces`. So an operator — or a process-engineer's new
branch — can never produce an artifact a downstream step cannot read. The type
system *is* the safety rail that makes agentic branch selection safe.

Immutability + versioning (critique point 13): an `SOP` is frozen. A promoted
branch does not mutate the running pipeline; it yields a *new* `SOP` at a new
version via `SOP.with_branch(...)`. This gives free rollback and a clean audit
trail of who changed the production procedure and when.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from .schemas import Artifact

log = logging.getLogger("asop.dag")

I = TypeVar("I", bound=Artifact)
O = TypeVar("O", bound=Artifact)


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #


@dataclass
class RunContext:
    """Threaded through every node call. Carries the knobs a node needs without
    making nodes reach into globals. `resources` is where a node asks for GPU
    tiles; `bus` is the inter-module channel; `extra` is campaign-specific."""

    round: int = 0
    dag_version: str = "v0"
    resources: "object | None" = None     # asop.modules.ResourceManager (optional)
    bus: "object | None" = None           # asop.bus.Bus (optional)
    logger: logging.Logger = field(default_factory=lambda: log)
    extra: dict = field(default_factory=dict)


# A node body: pure-ish function from (ctx, input artifact) -> output artifact.
NodeFn = Callable[[RunContext, Artifact], Artifact]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Node(Generic[I, O]):
    """A single step with exactly one implementation."""

    name: str
    consumes: type[Artifact]
    produces: type[Artifact]
    fn: NodeFn
    module: str = ""          # 'M3'..'M7', for resource accounting / bus routing
    description: str = ""      # one line the operator sees; the ONLY thing it reads about a branch

    def run(self, ctx: RunContext, inp: Artifact) -> Artifact:
        if not isinstance(inp, self.consumes):
            raise TypeError(
                f"step {self.name!r} consumes {self.consumes.__name__}, "
                f"got {type(inp).__name__}"
            )
        out = self.fn(ctx, inp)
        if not isinstance(out, self.produces):
            raise TypeError(
                f"step {self.name!r} must produce {self.produces.__name__}, "
                f"got {type(out).__name__}"
            )
        # Stamp provenance so downstream reconcile/steering can group by producer.
        return out.model_copy(update={"origin": f"{ctx.dag_version}:{self.name}"})


@dataclass(frozen=True)
class Fork:
    """A choice point. Every branch is a `Node` with the fork's exact
    consumes/produces, so the branches are interchangeable by construction.

    `allow_operator` is the README's safety gate: if False, the branch is chosen
    by the deterministic default (random, seeded) and no agent is consulted.
    A newly promoted branch starts with `allow_operator=True` only after it has
    passed the stretch-test and a human has opened that fork to the operator.
    """

    name: str
    consumes: type[Artifact]
    produces: type[Artifact]
    branches: dict[str, Node]
    allow_operator: bool = False
    default_branch: str | None = None     # used when the operator is not consulted
    module: str = ""

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError(f"fork {self.name!r} has no branches")
        for bname, b in self.branches.items():
            if b.consumes is not self.consumes or b.produces is not self.produces:
                raise TypeError(
                    f"branch {bname!r} of fork {self.name!r} has type "
                    f"{b.consumes.__name__}->{b.produces.__name__}, fork requires "
                    f"{self.consumes.__name__}->{self.produces.__name__}"
                )
        if self.default_branch is not None and self.default_branch not in self.branches:
            raise ValueError(
                f"fork {self.name!r} default_branch {self.default_branch!r} is not a branch"
            )


Step = Node | Fork


# --------------------------------------------------------------------------- #
# SOP
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SOP:
    """An immutable, versioned, type-checked pipeline."""

    version: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("an SOP needs at least one step")
        prev: type[Artifact] | None = None
        for s in self.steps:
            if prev is not None and s.consumes is not prev:
                raise TypeError(
                    f"step {s.name!r} consumes {s.consumes.__name__} but the previous "
                    f"step produces {prev.__name__}: the pipeline does not chain"
                )
            prev = s.produces

    # -- introspection ------------------------------------------------------- #

    @property
    def consumes(self) -> type[Artifact]:
        return self.steps[0].consumes

    @property
    def produces(self) -> type[Artifact]:
        return self.steps[-1].produces

    def index_of(self, step_name: str) -> int:
        for i, s in enumerate(self.steps):
            if s.name == step_name:
                return i
        raise KeyError(f"no step named {step_name!r} in SOP {self.version}")

    def sub(self, start: int, end: int | None = None) -> "SOP":
        """A contiguous slice of the pipeline, as its own SOP. This is what the
        process-engineer stretch-test runs: ``sop.sub(max(i-2, 0))``."""
        chunk = self.steps[start:end]
        return SOP(version=f"{self.version}:sub[{start}:{end if end is not None else len(self.steps)}]",
                   steps=tuple(chunk))

    # -- immutable mutation: returns a NEW SOP ------------------------------- #

    def with_branch(self, step_name: str, branch_name: str, branch: Node,
                    *, new_version: str, allow_operator: bool = True) -> "SOP":
        """Return a new SOP where `step_name` gains `branch`. If the named step is
        a plain Node it is widened into a Fork whose default is the original
        implementation. The running SOP is never mutated (critique point 13)."""
        i = self.index_of(step_name)
        s = self.steps[i]
        if isinstance(s, Fork):
            if branch_name in s.branches:
                raise ValueError(f"fork {step_name!r} already has a branch {branch_name!r}")
            new_fork = replace(s, branches={**s.branches, branch_name: branch},
                               allow_operator=allow_operator or s.allow_operator)
        else:  # widen a Node into a Fork keeping the original as default
            new_fork = Fork(
                name=s.name, consumes=s.consumes, produces=s.produces,
                branches={s.name: _node_as_branch(s), branch_name: branch},
                allow_operator=allow_operator, default_branch=s.name, module=s.module,
            )
        steps = list(self.steps)
        steps[i] = new_fork
        return SOP(version=new_version, steps=tuple(steps))


def _node_as_branch(node: Node) -> Node:
    """A Node used as a fork's default branch keeps its own fn but is renamed to
    the generic 'baseline' slot handled above; here we just return it unchanged."""
    return node


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


@dataclass
class TraceEntry:
    step: str
    branch: str | None
    chosen_by: str            # 'node' | 'operator' | 'default' | 'random'
    rationale: str
    produced: str             # artifact type name


# A branch selector: given (ctx, fork) -> (branch_name, chosen_by, rationale).
Selector = Callable[[RunContext, Fork], tuple[str, str, str]]


class Executor:
    """Walks an SOP, threading the artifact and recording a provenance trace.

    The only place an agent can influence control flow is `select` at an allowed
    fork. Everything else is deterministic Python.
    """

    def __init__(self, select: Selector | None = None):
        # Default selector: deterministic. Honors fork.default_branch; if none,
        # picks the first branch. A real campaign passes operator.select here.
        self.select = select or _default_select

    def run(self, sop: SOP, initial: Artifact, ctx: RunContext | None = None
            ) -> tuple[Artifact, list[TraceEntry]]:
        ctx = ctx or RunContext(dag_version=sop.version)
        ctx = dataclasses.replace(ctx, dag_version=sop.version)
        if not isinstance(initial, sop.consumes):
            raise TypeError(
                f"SOP {sop.version} consumes {sop.consumes.__name__}, "
                f"got {type(initial).__name__}"
            )
        art = initial
        trace: list[TraceEntry] = []
        for step in sop.steps:
            if isinstance(step, Node):
                art = step.run(ctx, art)
                trace.append(TraceEntry(step.name, None, "node", "", type(art).__name__))
            else:  # Fork
                bname, chosen_by, why = self._choose(ctx, step)
                branch = step.branches[bname]
                art = branch.run(ctx, art)
                trace.append(TraceEntry(step.name, bname, chosen_by, why, type(art).__name__))
            ctx.logger.info("SOP %s: %s -> %s", sop.version, trace[-1].step, trace[-1].produced)
        return art, trace

    def _choose(self, ctx: RunContext, fork: Fork) -> tuple[str, str, str]:
        if not fork.allow_operator:
            # Closed fork: deterministic default, no agent consulted.
            return _default_select(ctx, fork)
        name, chosen_by, why = self.select(ctx, fork)
        if name not in fork.branches:
            raise ValueError(
                f"selector picked branch {name!r} not in fork {fork.name!r} "
                f"(have {sorted(fork.branches)})"
            )
        return name, chosen_by, why


def _default_select(ctx: RunContext, fork: Fork) -> tuple[str, str, str]:
    if fork.default_branch is not None:
        return fork.default_branch, "default", "fork default_branch"
    first = next(iter(fork.branches))
    return first, "default", "first branch (no default set)"
