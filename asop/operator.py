"""The operator: the *only* agent allowed to influence SOP control flow, and
only by naming a branch at a fork that a human has opened (`allow_operator`).

Separation of powers (README critique point 9 & 13):
  * the operator SELECTS among existing, type-compatible branches — it can never
    add, remove, or edit a branch, nor touch any other step;
  * a process-engineer PROPOSES branches but cannot select them in production;
  * promotion (opening a fork to the operator) is a human/gated action.

The operator needs an *objective* to choose well (critique point 9). The default
policy here is an explicit, auditable ε-greedy bandit over each branch's
historical campaign-objective — no LLM required, fully runnable, and a sane
baseline the README lacked. An LLM operator can be dropped in via the same
`Selector` signature (`select(ctx, fork) -> (branch, chosen_by, rationale)`);
it should return a branch name plus a rationale that is logged into the trace.

Every selection is logged with who chose and why, so the campaign can later ask
the question the README poses honestly: did opening this fork to the operator
actually beat the random baseline?
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .dag import Fork, RunContext
from .schemas import ForkChoice

log = logging.getLogger("asop.operator")


@dataclass
class BranchStats:
    pulls: int = 0
    total_objective: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_objective / self.pulls if self.pulls else 0.0


@dataclass
class BanditOperator:
    """ε-greedy over branch history, scoped per fork. Deterministic given `seed`.

    `history_path` persists the per-branch running means across rounds so the
    operator's knowledge survives a campaign restart (as loamlite persists Elo).
    """

    epsilon: float = 0.2
    seed: int = 0
    history_path: Path | None = None
    _stats: dict[str, dict[str, BranchStats]] = field(default_factory=lambda: defaultdict(dict))
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.history_path and self.history_path.is_file():
            self._load()

    # -- the Selector interface the Executor calls -------------------------- #

    def select(self, ctx: RunContext, fork: Fork) -> tuple[str, str, str]:
        names = sorted(fork.branches)
        stats = self._stats[fork.name]
        unseen = [n for n in names if n not in stats or stats[n].pulls == 0]
        if unseen:
            pick = self._rng.choice(unseen)
            return pick, "operator", f"explore: {pick!r} has no history yet"
        if self._rng.random() < self.epsilon:
            pick = self._rng.choice(names)
            return pick, "operator", f"ε-explore (ε={self.epsilon}) among {names}"
        pick = max(names, key=lambda n: stats[n].mean)
        means = {n: round(stats[n].mean, 4) for n in names}
        return pick, "operator", f"exploit best mean objective {means}"

    # -- feedback: call after M4 produces the round's objective ------------- #

    def record(self, fork_name: str, branch: str, objective: float) -> None:
        s = self._stats[fork_name].setdefault(branch, BranchStats())
        s.pulls += 1
        s.total_objective += objective
        if self.history_path:
            self._save()

    def as_choice(self, ctx: RunContext, fork: Fork) -> ForkChoice:
        branch, _by, why = self.select(ctx, fork)
        exp = self._stats[fork.name].get(branch)
        return ForkChoice(branch=branch, rationale=why,
                          expected_objective=exp.mean if exp and exp.pulls else None)

    # -- persistence -------------------------------------------------------- #

    def _save(self) -> None:
        assert self.history_path is not None
        blob = {fork: {b: [s.pulls, s.total_objective] for b, s in bs.items()}
                for fork, bs in self._stats.items()}
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=1))
        tmp.replace(self.history_path)

    def _load(self) -> None:
        assert self.history_path is not None
        blob = json.loads(self.history_path.read_text())
        for fork, bs in blob.items():
            for b, (pulls, total) in bs.items():
                self._stats[fork][b] = BranchStats(pulls=pulls, total_objective=total)


@dataclass
class LLMOperator:
    """An LLM-backed operator with a deliberately narrow interface: it is given NO
    skills, NO execution tools, and NO code — only, for each branch, its name,
    one-line description, and running mean objective — and must return one of the
    branch names. That narrowness is the point of the operator/engineer split:
    branch selection is a closed multiple-choice, not skill-following or
    code-generation, so the production loop does not depend on run-time code
    authoring and can be served by a smaller or cheaper model when that is the
    right trade-off.

    `decide(prompt, choices) -> name` is any callable (a pydantic-ai agent with a
    `Literal[...]`/enum output, a local vLLM call, even a human). The prompt below
    is all it ever sees; `fallback` (default: the bandit) covers a malformed or
    out-of-set answer so an off-contract response can never stall or corrupt the DAG.
    """

    decide: "Callable[[str, list[str]], str]"
    stats: "BanditOperator | None" = None     # supplies objective history + fallback
    fallback_seed: int = 0

    def __post_init__(self) -> None:
        self._fallback = self.stats or BanditOperator(seed=self.fallback_seed)

    def select(self, ctx: RunContext, fork: Fork) -> tuple[str, str, str]:
        names = sorted(fork.branches)
        means = self._fallback._stats.get(fork.name, {})
        lines = []
        for n in names:
            desc = fork.branches[n].description or "(no description)"
            m = means.get(n)
            hist = f"mean objective {m.mean:.3f} over {m.pulls}" if m and m.pulls else "no history yet"
            lines.append(f"- {n}: {desc}  [{hist}]")
        prompt = (
            f"Pick the single best branch to run next at fork '{fork.name}'. "
            f"Reply with exactly one branch name from the list.\n" + "\n".join(lines))
        try:
            pick = (self.decide(prompt, names) or "").strip()
        except Exception as exc:  # an erroring decide() falls back, never crashes the loop
            fb, _by, why = self._fallback.select(ctx, fork)
            return fb, "operator", f"LLM errored ({exc!r}); fell back to bandit: {why}"
        if pick not in fork.branches:
            fb, _by, why = self._fallback.select(ctx, fork)
            return fb, "operator", f"LLM returned {pick!r} (not a branch); fell back: {why}"
        return pick, "operator", "LLM choice among branch descriptions + objective history"

    def record(self, fork_name: str, branch: str, objective: float) -> None:
        self._fallback.record(fork_name, branch, objective)


@dataclass
class RandomOperator:
    """The README's stated fallback for a fork not opened to the operator, exposed
    as a Selector for A/B comparison against BanditOperator."""

    seed: int = 0
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(self, ctx: RunContext, fork: Fork) -> tuple[str, str, str]:
        pick = self._rng.choice(sorted(fork.branches))
        return pick, "random", "uniform random (fork not under operator policy)"
