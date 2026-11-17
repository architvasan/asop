"""Registry adapters: module-standard glue (store-query + result-parsing) that
turns a frozen `BranchRecord.script` into a deterministic `dag.Node`.

An adapter is written once per step kind; a process-engineer then only writes the
scientific `script`. Each submodule exposes an `Adapter` plus a `baseline_record()`
that seeds the registry with the module's default branch.
"""

from __future__ import annotations

__all__ = ["m4_qc", "m4_qc_energy"]
