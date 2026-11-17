"""The M1..M7 module registry and the node (resource) arbiter.

This file is where asop names its modules *once*, consistently, and maps each to
(a) the loamlite machinery it reuses and (b) its node budget. It replaces the
README's off-by-one "details" sections (critique point 2) with a single source
of truth.

Nothing here reimplements loamlite: M1 and M2 ARE loamlite (literature/RAG
skills + the Jnana tournament/validation loop); M3..M7 are loamlite *skills*
invoked as DAG-node bodies. asop only owns the control layer above them.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger("asop.modules")


@dataclass(frozen=True)
class Module:
    code: str          # 'M1'..'M7'
    name: str
    loamlite: str      # which loamlite capability realizes it
    engineers: tuple[str, ...] = ()   # process-engineer roles (M3..M6 only)
    node_budget: int = 0              # nodes reserved for this module's PE sandbox


# Single, consistent catalogue. Note the fixes vs. the README:
#   * recon (target structure prediction + MD) is M5, not M4;
#   * optimization (DPO/KTO/GRPO, preference splitting) is M6, not M5;
#   * M4 is purely QC/evaluation.
REGISTRY: dict[str, Module] = {
    "M1": Module("M1", "Knowledge gathering", "loamlite skills: fetch-literature, rag-build-index, rag-query"),
    "M2": Module("M2", "Hypothesis generation", "loamlite tournament: Jnana generation+ranking, validator/reviewer"),
    "M3": Module("M3", "Binder design", "loamlite skills: rfdiffusion3, proteinmpnn, esm/plm, chai/boltz",
                 engineers=("design", "sequence"), node_budget=2),
    "M4": Module("M4", "Quality control / evaluation", "loamlite skills: chai/boltz (ipSAE), pyrosetta, developability, md-simulation",
                 engineers=("evaluation",), node_budget=2),
    "M5": Module("M5", "Target reconnaissance", "loamlite skills: chai (holo/PTM/cofactor), bioemu, molecular-simulations (MSM)",
                 engineers=("structure", "md-sim", "md-analysis"), node_budget=6),
    "M6": Module("M6", "Optimization", "loamlite skills: diversity-dpo-mpnn, selectivity-dpo, GRPO/KTO",
                 engineers=("preference", "optimizer"), node_budget=1),
    "M7": Module("M7", "Mechanistic understanding", "loamlite skills: deepdrivewe-koff, msm kinetics, CALVADOS, free-energy"),
}


# --------------------------------------------------------------------------- #
# Resource arbiter
# --------------------------------------------------------------------------- #


@dataclass
class ResourceManager:
    """Arbitrates GPU nodes between the *production* SOP and the process-engineer
    sandboxes (critique point 11). A PE smoke/stretch reservation can never
    starve production below `production_floor`.

    This is deliberately a thin, in-memory counting arbiter with the right shape
    to be backed by loamlite's `execution.hpc` pool (`pool_status`/`run_batch`);
    the counts here mirror what that pool reports.
    """

    total_nodes: int
    production_floor: int = 0          # nodes production always keeps
    _reserved: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def free(self) -> int:
        return self.total_nodes - sum(self._reserved.values())

    def reserve(self, owner: str, nodes: int, *, production: bool = False) -> bool:
        """Try to reserve `nodes` for `owner`. PE (non-production) requests are
        refused if granting them would push production below its floor."""
        with self._lock:
            if nodes <= 0:
                return True
            if not production and self.free - nodes < self.production_floor:
                log.info("deny %s +%d nodes: would breach production floor %d (free=%d)",
                         owner, nodes, self.production_floor, self.free)
                return False
            if nodes > self.free:
                return False
            self._reserved[owner] = self._reserved.get(owner, 0) + nodes
            return True

    def release(self, owner: str, nodes: int | None = None) -> None:
        with self._lock:
            held = self._reserved.get(owner, 0)
            give = held if nodes is None else min(nodes, held)
            if held - give <= 0:
                self._reserved.pop(owner, None)
            else:
                self._reserved[owner] = held - give

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"total": self.total_nodes, "free": self.free, **self._reserved}
