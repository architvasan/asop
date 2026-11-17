"""Typed artifacts that flow on the SOP-DAG edges, and the meta-objects the
operator and process-engineers exchange.

Every artifact is a strict pydantic model (``extra='forbid'``) exactly as
loamlite's agent outputs are: an edge whose producer emits an unknown key, or
whose consumer is handed the wrong type, fails loudly instead of silently
corrupting a downstream step. This is the data contract the README was missing
(critique point 7): two branches at a fork are interchangeable *iff* they share
``consumes``/``produces`` artifact types.

Carrying a scalar ``objective`` on the evaluation artifacts is what makes the
process-engineer stretch-test (critique point 8) a real non-regression gate
rather than a vibe.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    """Base for everything that travels on a DAG edge."""

    model_config = ConfigDict(extra="forbid")

    # Provenance breadcrumb: which (dag_version, step, branch) emitted this.
    # Set by the executor, not by module code. Lets reconcile/steering group
    # evidence by producing run exactly as loamlite groups claims by run id.
    origin: str = ""


# --------------------------------------------------------------------------- #
# M3 / M4 / M5 / M6 / M7 domain artifacts
# --------------------------------------------------------------------------- #


class TargetSpec(Artifact):
    """What a design round is aimed at. Produced by M5 (reconnaissance), or
    seeded by a human for round 0."""

    name: str
    sequence: str
    structure_path: str | None = None          # holo structure to design against
    hotspots: list[int] = Field(default_factory=list)   # 1-indexed target residues
    cofactors: list[str] = Field(default_factory=list)  # e.g. ['FE2', 'AKG']
    epitope_is_disordered: bool = False
    off_targets: list[str] = Field(default_factory=list)  # paralog seqs for selectivity


class Design(Artifact):
    """One candidate binder."""

    design_id: str
    sequence: str
    backbone_path: str | None = None   # de novo backbone (CIF/PDB) if any
    method: str = ""                   # which M3 branch made it


class DesignBatch(Artifact):
    """M3 output: a pool of candidate binders for one target."""

    target: TargetSpec
    designs: list[Design]

    @property
    def size(self) -> int:
        return len(self.designs)


class Metric(Artifact):
    """One scored quantity on one design (ipSAE, dG_bind, developability, ...)."""

    design_id: str
    name: str
    value: float
    gate_pass: bool | None = None      # if this metric was used as a hard gate


class QCReport(Artifact):
    """M4 output: metrics per design plus a single campaign-objective scalar used
    for non-regression comparisons between branches."""

    target_name: str
    metrics: list[Metric]
    survivors: list[str] = Field(default_factory=list)   # design_ids that passed all gates
    objective: float = 0.0             # higher is better; the thing asop maximizes
    objective_definition: str = ""     # human-readable: what `objective` counts


class ReconEnsemble(Artifact):
    """M5 output: structures / conformational ensemble of the target and the
    recommended design target(s) drawn from it."""

    target_name: str
    ensemble_paths: list[str] = Field(default_factory=list)
    recommended_targets: list[TargetSpec] = Field(default_factory=list)
    note: str = ""


class PreferenceSet(Artifact):
    """M6 input: preference pairs mined from (designs, QC) for DPO/KTO/GRPO."""

    target_name: str
    pairs: list[dict[str, Any]] = Field(default_factory=list)   # {chosen, rejected, margin}
    scheme: Literal["dpo", "kto", "grpo"] = "dpo"


class ModelDelta(Artifact):
    """M6 output: a pointer to updated generator weights for M3 to pick up."""

    base_model: str
    weights_path: str
    train_metric: float = 0.0
    note: str = ""


class MechInsight(Artifact):
    """M7 output: a grounded mechanistic finding that may flow *backward* to M2.
    (Critique point 12: this backward edge must exist.)"""

    target_name: str
    statement: str
    evidence_paths: list[str] = Field(default_factory=list)
    changes_scoring: bool = False      # True => M4 should reweight its objective


# --------------------------------------------------------------------------- #
# Meta objects: operator decisions and process-engineer proposals
# --------------------------------------------------------------------------- #


class ForkChoice(BaseModel):
    """What the operator returns at an allowed fork. It may only *name* a branch;
    it cannot alter the DAG (critique point 13: separation of powers)."""

    model_config = ConfigDict(extra="forbid")

    branch: str
    rationale: str = ""
    expected_objective: float | None = None


class BranchProposal(BaseModel):
    """What a process-engineer submits. A proposal is NOT trusted until it passes
    `engineer.evaluate` (smoke + stretch, non-regression)."""

    model_config = ConfigDict(extra="forbid")

    module: str                 # 'M3' | 'M4' | 'M5' | 'M6'
    step_name: str              # the step/fork this branch attaches to
    branch_name: str
    description: str = ""


class PromotionResult(BaseModel):
    """Outcome of evaluating a proposal. ``accepted`` branches become a new,
    higher-versioned SOP; nothing mutates in place."""

    model_config = ConfigDict(extra="forbid")

    proposal: BranchProposal
    accepted: bool
    smoke_ok: bool
    stretch_ok: bool
    baseline_objective: float
    candidate_objective: float
    reason: str = ""
    new_dag_version: str | None = None
