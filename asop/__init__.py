"""asop — Agentic Standard Operating Procedure.

A control layer on top of `loamlite`. loamlite supplies the hypothesis loop
(Jnana tournament + typed validator/code-generator/reviewer agents), the HPC
execution layer, and the scientific skills. asop adds:

  * a versioned, immutable **SOP-DAG** of typed steps (M3..M6 production loop),
  * an **operator** agent that may only *select* a branch at an allowed fork,
  * a **process-engineer** meta-loop that may *propose* new branches and must
    pass a strict stretch-test gate before a branch is promoted into a new DAG
    version,
  * a typed **module bus** for the forward/backward agent communication network.

Design invariant (inherited from loamlite): the orchestration is plain Python;
agents return strict typed objects; capability is granted by which tools/forks a
role is handed, never policed at runtime.
"""

from __future__ import annotations

__all__ = [
    "dag",
    "schemas",
    "operator",
    "engineer",
    "registry",
    "modules",
    "bus",
    "store",
    "batch",
    "nodes",
]

__version__ = "0.0.1"
