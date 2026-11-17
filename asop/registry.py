"""The DAG registry: the append-only catalogue of real, frozen branch code that
process-engineers commit to once a branch passes the stretch gate.

The flow the campaign runs on:

    process-engineer (uses skills to author code)
        writes a real batch SCRIPT  ──►  engineer.evaluate (smoke + stretch)
        proven non-regressing?       ──►  registry.register(BranchRecord)   [commit]
                                            │
    operator (closed-choice interface)  ──►  Registry.build_sop()  assembles the SOP
        selects a branch NAME             from every registered branch, and runs it.

What is serializable — and therefore what the registry stores — is the *script*
(the real scientific code the engineer wrote), plus the name of an `Adapter` that
knows how to wrap any script for that step into a deterministic `batch_node`
(the store query + result parsing are module-standard, not per-branch). So a
branch record is fully reproducible from disk, and the registry is the single
source of truth for what the production DAG currently is.

Append-only + versioned: committing a branch never edits an existing record, so
`build_sop` can reconstruct any past DAG and rollback is just "build from the
records up to version N".
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict

from .dag import SOP, Fork, Node
from .schemas import Artifact, PromotionResult

log = logging.getLogger("asop.registry")


class BranchRecord(BaseModel):
    """One registered branch: the real code plus how to build and trust it."""

    model_config = ConfigDict(extra="forbid")

    module: str
    step: str
    branch_name: str
    adapter: str                      # which Adapter materializes this branch
    description: str = ""
    script: str                       # the real code the engineer wrote
    venv_path: str | None = None
    # provenance (filled at commit from the PromotionResult)
    proposed_by: str = ""
    promoted_version: str = ""
    baseline_objective: float | None = None
    candidate_objective: float | None = None
    is_baseline: bool = False         # the original production branch for its step
    ts: float = 0.0


@dataclass
class Adapter:
    """Module-standard glue that turns a `BranchRecord.script` into a `Node`.

    One adapter per step kind (e.g. an "M4-qc" adapter). It owns the store query
    and result parsing — the deterministic plumbing — so an engineer only has to
    write the scientific `script`. `make_node(record) -> Node`."""

    name: str
    consumes: type[Artifact]
    produces: type[Artifact]
    make_node: Callable[[BranchRecord], Node]


@dataclass
class Registry:
    """Append-only branch catalogue on disk (JSONL), plus SOP assembly."""

    path: Path
    adapters: dict[str, Adapter] = field(default_factory=dict)
    _records: list[BranchRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self._records = [BranchRecord.model_validate_json(line)
                             for line in self.path.read_text().splitlines() if line.strip()]

    def add_adapter(self, adapter: Adapter) -> None:
        self.adapters[adapter.name] = adapter

    # -- commit ------------------------------------------------------------- #

    def register(self, record: BranchRecord) -> BranchRecord:
        """Commit a branch. Rejects a duplicate (step, branch_name) and an unknown
        adapter — the registry is the trust boundary, so it validates on write."""
        if record.adapter not in self.adapters:
            raise KeyError(f"unknown adapter {record.adapter!r}; add_adapter first")
        if any(r.step == record.step and r.branch_name == record.branch_name for r in self._records):
            raise ValueError(f"branch {record.branch_name!r} already registered on step {record.step!r}")
        # type-check the script materializes to a node with the adapter's I/O
        node = self.adapters[record.adapter].make_node(record)
        if node.consumes is not self.adapters[record.adapter].consumes or \
           node.produces is not self.adapters[record.adapter].produces:
            raise TypeError("adapter produced a node with mismatched I/O types")
        record = record.model_copy(update={"ts": record.ts or time.time()})
        self._records.append(record)
        with self.path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        log.info("registered %s.%s on step %s (%s)", record.module, record.branch_name,
                 record.step, "baseline" if record.is_baseline else "promoted")
        return record

    def commit_promotion(self, result: PromotionResult, *, adapter: str, script: str,
                         description: str = "", proposed_by: str = "",
                         venv_path: str | None = None) -> BranchRecord:
        """Turn an accepted `engineer.evaluate` result into a committed branch.
        Refuses to commit a rejected proposal — the gate is enforced here, not
        by trust."""
        if not result.accepted:
            raise ValueError(f"cannot register a rejected proposal: {result.reason}")
        p = result.proposal
        rec = BranchRecord(
            module=p.module, step=p.step_name, branch_name=p.branch_name, adapter=adapter,
            description=description, script=script, venv_path=venv_path, proposed_by=proposed_by,
            promoted_version=result.new_dag_version or "",
            baseline_objective=result.baseline_objective,
            candidate_objective=result.candidate_objective)
        return self.register(rec)

    # -- query / assembly --------------------------------------------------- #

    def branches(self, step: str) -> list[BranchRecord]:
        return [r for r in self._records if r.step == step]

    def materialize(self, record: BranchRecord) -> Node:
        return self.adapters[record.adapter].make_node(record)

    def build_step(self, step: str, *, allow_operator: bool = True) -> Fork:
        """Assemble one step from its registered branches as a Fork named by the
        STEP (so the step is addressable by name regardless of how many branches
        it has). A single-branch fork is closed and just runs its one branch;
        `allow_operator` only takes effect once there is a real choice."""
        recs = self.branches(step)
        if not recs:
            raise KeyError(f"no branches registered for step {step!r}")
        nodes = {r.branch_name: self.materialize(r) for r in recs}
        baseline = next((r.branch_name for r in recs if r.is_baseline), None)
        default = baseline or (next(iter(nodes)) if len(nodes) == 1 else None)
        adapter = self.adapters[recs[0].adapter]
        return Fork(name=step, consumes=adapter.consumes, produces=adapter.produces,
                    branches=nodes, allow_operator=allow_operator and len(nodes) > 1,
                    default_branch=default, module=recs[0].module)

    def build_sop(self, version: str, step_order: list[str],
                  *, allow_operator: bool = True) -> SOP:
        """The current production SOP, assembled from the registry in the given
        step order. This is what the operator runs; it always reflects the latest
        committed branches."""
        steps = tuple(self.build_step(s, allow_operator=allow_operator) for s in step_order)
        return SOP(version=version, steps=steps)
