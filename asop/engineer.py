"""The process-engineer meta-loop: propose a new branch, prove it, promote it.

This is the README's §"Process engineers" made concrete and given a real gate
(critique point 8). A process engineer owns one module (M3..M6), gets a small
node budget to play in, and may *propose* a branch. A proposal becomes part of
the production SOP only after:

  1. **smoke test** — the branch runs on one probe artifact and PRODUCES an
     artifact of the fork's declared output type (type contract, cheap);
  2. **stretch test** — the README's own rule: splice the branch in and run the
     downstream stretch ``sop.sub(max(i-2, 0) .. end)`` on a held-out probe set,
     to completion, and require the end **objective** to be NON-REGRESSING versus
     the current production branch on the same probes (not just "it ran");
  3. **promotion** — only then do we emit a NEW, higher-versioned SOP with the
     branch added (immutable; the running SOP is untouched). Opening the fork to
     the operator (`allow_operator=True`) is the human-gated step; `evaluate`
     returns the candidate SOP but does not deploy it.

The engineer never edits the DAG in place and never selects branches in
production — that is the operator's job (separation of powers, critique 13).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .dag import SOP, Executor, Fork, Node, RunContext
from .schemas import Artifact, BranchProposal, PromotionResult, QCReport

log = logging.getLogger("asop.engineer")


def _objective_of(art: Artifact) -> float:
    """Pull the campaign objective off a terminal artifact. The production SOP
    ends in a QCReport; a sub-pipeline may end elsewhere, so fall back to 0 and
    let the caller supply a scorer for non-QC tails."""
    if isinstance(art, QCReport):
        return art.objective
    return float(getattr(art, "objective", 0.0) or 0.0)


@dataclass
class ProbeSet:
    """The held-out inputs the stretch-test runs on. These must be *inputs to the
    stretch's first step* (i.e. of type ``sop.sub(start).consumes``), so a probe
    is produced by replaying production up to that cut, or cached from a prior
    round. Keeping them fixed is what makes the non-regression comparison fair."""

    inputs: list[Artifact]
    scorer = staticmethod(_objective_of)   # terminal-artifact -> float


@dataclass
class ProcessEngineer:
    """Owns one module. `node_budget` is advisory here (the ResourceManager
    enforces it); it documents the README's per-engineer allocation."""

    module: str                 # 'M3'..'M6'
    node_budget: int = 2
    epsilon: float = 0.0        # non-regression slack: accept if cand >= base - epsilon
    executor: Executor = field(default_factory=Executor)

    # -- step 1: smoke -------------------------------------------------------- #

    def smoke(self, branch: Node, probe: Artifact, ctx: RunContext) -> tuple[bool, str]:
        try:
            out = branch.run(ctx, probe)
        except Exception as exc:  # a branch that throws is simply rejected
            return False, f"branch raised: {exc!r}"
        if not isinstance(out, branch.produces):
            return False, f"produced {type(out).__name__}, expected {branch.produces.__name__}"
        return True, f"produced {type(out).__name__}"

    # -- step 2: stretch (the README's max(i-2,0)->end) ---------------------- #

    def stretch(self, sop: SOP, step_name: str, branch: Node, probes: ProbeSet,
                ctx: RunContext) -> tuple[bool, float, float, str]:
        """Run the downstream stretch with the current production branch vs. the
        candidate, on the same probes, and compare terminal objectives."""
        i = sop.index_of(step_name)
        start = max(i - 2, 0)
        stretch_sop = sop.sub(start)
        # The probe must feed the stretch's first step.
        want = stretch_sop.consumes
        for p in probes.inputs:
            if not isinstance(p, want):
                return (False, 0.0, 0.0,
                        f"probe type {type(p).__name__} != stretch input {want.__name__}")

        # Baseline SOP for the stretch: the production branch at step i.
        candidate_sop = stretch_sop.with_branch(
            step_name, branch.name, branch, new_version=f"{sop.version}+cand", allow_operator=False)
        # Force the stretch to USE the candidate branch deterministically.
        candidate_sop = _pin_fork(candidate_sop, step_name, branch.name)
        baseline_sop = _pin_to_default(stretch_sop, step_name)

        base = self._mean_objective(baseline_sop, probes, ctx)
        cand = self._mean_objective(candidate_sop, probes, ctx)
        ok = cand >= base - self.epsilon
        reason = (f"stretch[{start}:] mean objective  baseline={base:.4f}  "
                  f"candidate={cand:.4f}  (slack ε={self.epsilon})  -> "
                  f"{'non-regressing' if ok else 'REGRESSION'}")
        return ok, base, cand, reason

    def _mean_objective(self, sop: SOP, probes: ProbeSet, ctx: RunContext) -> float:
        vals = []
        for p in probes.inputs:
            out, _trace = self.executor.run(sop, p, ctx)
            vals.append(probes.scorer(out))
        return sum(vals) / len(vals) if vals else 0.0

    # -- orchestration: propose -> (smoke, stretch) -> PromotionResult ------- #

    def evaluate(self, sop: SOP, proposal: BranchProposal, branch: Node,
                 probes: ProbeSet, ctx: RunContext) -> tuple[PromotionResult, SOP | None]:
        """Full gate. Returns (result, candidate_sop_or_None). The candidate SOP,
        when accepted, has the fork OPEN to the operator but is NOT deployed here
        — a human promotes it by making it the active SOP."""
        if proposal.module != self.module:
            return (self._fail(proposal, False, False, 0, 0,
                               f"engineer owns {self.module}, proposal targets {proposal.module}"),
                    None)
        probe0 = probes.inputs[0] if probes.inputs else None
        if probe0 is None:
            return self._fail(proposal, False, False, 0, 0, "no probes supplied"), None

        smoke_ok, smoke_why = self.smoke(branch, probe0, ctx)
        if not smoke_ok:
            return self._fail(proposal, False, False, 0, 0, f"smoke failed: {smoke_why}"), None

        stretch_ok, base, cand, why = self.stretch(sop, proposal.step_name, branch, probes, ctx)
        if not stretch_ok:
            return self._fail(proposal, True, False, base, cand, why), None

        new_version = _bump(sop.version)
        candidate_sop = sop.with_branch(
            proposal.step_name, proposal.branch_name, branch,
            new_version=new_version, allow_operator=True)
        result = PromotionResult(
            proposal=proposal, accepted=True, smoke_ok=True, stretch_ok=True,
            baseline_objective=base, candidate_objective=cand,
            reason=f"smoke ok; {why}", new_dag_version=new_version)
        log.info("ACCEPTED %s.%s on %s: %s", proposal.module, proposal.branch_name,
                 proposal.step_name, result.reason)
        return result, candidate_sop

    @staticmethod
    def _fail(proposal, smoke, stretch, base, cand, reason) -> PromotionResult:
        log.info("REJECTED %s.%s: %s", proposal.module, proposal.branch_name, reason)
        return PromotionResult(
            proposal=proposal, accepted=False, smoke_ok=smoke, stretch_ok=stretch,
            baseline_objective=base, candidate_objective=cand, reason=reason)


# --------------------------------------------------------------------------- #
# helpers: pin a fork to one branch for a deterministic A/B
# --------------------------------------------------------------------------- #


def _pin_fork(sop: SOP, step_name: str, branch_name: str) -> SOP:
    """Return a copy of `sop` where the named fork is closed to the operator and
    defaulted to `branch_name`, so the executor runs exactly that branch."""
    from dataclasses import replace
    steps = list(sop.steps)
    i = sop.index_of(step_name)
    s = steps[i]
    if isinstance(s, Fork):
        steps[i] = replace(s, allow_operator=False, default_branch=branch_name)
    return SOP(version=f"{sop.version}#pin={branch_name}", steps=tuple(steps))


def _pin_to_default(sop: SOP, step_name: str) -> SOP:
    """Pin the fork to its current default (the production branch)."""
    from dataclasses import replace
    steps = list(sop.steps)
    i = sop.index_of(step_name)
    s = steps[i]
    if isinstance(s, Fork):
        default = s.default_branch or next(iter(s.branches))
        steps[i] = replace(s, allow_operator=False, default_branch=default)
    return SOP(version=f"{sop.version}#baseline", steps=tuple(steps))


def _bump(version: str) -> str:
    """v3 -> v4; v3.2 -> v3.3; anything else -> <version>+1."""
    head = version.split(":", 1)[0].split("#", 1)[0]
    if head.startswith("v") and head[1:].replace(".", "").isdigit():
        parts = head[1:].split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return "v" + ".".join(parts)
    return head + "+1"
