"""Factory for production DAG nodes: real, deterministic functions that read the
store and submit an HPC batch. These are the ONLY things an operator chooses
between — it never sees the code inside, only the branch name.

A `batch_node` turns four plain Python callables into a `dag.Node`:

  * `build_rows(store, inp) -> list[row]`  — the deterministic work query
    (e.g. "designs for this target awaiting QC"). No LLM.
  * a `script` string               — the one script run on every row, per the
    `batch.BatchRunner` contract (argv: input.jsonl output.jsonl).
  * `parse(receipt, inp, store) -> produces` — fold per-row results back into
    the typed output artifact and write them to the store.

The operator's prompt for a fork over these is tiny: "here are N branch names,
each a one-line description and its mean objective so far — return one name."
Because there is no skill to follow and no code to generate at selection time,
the production loop does not depend on run-time code authoring and can be served
by whatever model size fits the budget.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .batch import BatchReceipt, BatchRunner
from .dag import Node, RunContext
from .schemas import Artifact
from .store import Store

log = logging.getLogger("asop.nodes")

BuildRows = Callable[[Store, Artifact], list[dict[str, Any]]]
ParseResults = Callable[[BatchReceipt, Artifact, Store], Artifact]


def get_store(ctx: RunContext) -> Store:
    store = ctx.extra.get("store")
    if not isinstance(store, Store):
        raise RuntimeError("batch node needs a Store in ctx.extra['store']")
    return store


def get_runner(ctx: RunContext) -> BatchRunner:
    runner = ctx.extra.get("runner")
    if runner is None:
        raise RuntimeError("batch node needs a BatchRunner in ctx.extra['runner']")
    return runner


def batch_node(
    *,
    name: str,
    module: str,
    consumes: type[Artifact],
    produces: type[Artifact],
    script: str,
    build_rows: BuildRows,
    parse: ParseResults,
    description: str = "",
    venv_path: str | None = None,
    resources: dict | None = None,
) -> Node:
    """Build a deterministic HPC-backed node. This is what a user (or Claude)
    writes once and freezes; the operator only ever names it."""

    def fn(ctx: RunContext, inp: Artifact) -> Artifact:
        store = get_store(ctx)
        runner = get_runner(ctx)
        rows = build_rows(store, inp)
        if not rows:
            log.info("node %s: no work rows for this input; passing through empty result", name)
        receipt = runner.submit(step=name, script=script, rows=rows,
                                venv_path=venv_path, resources=resources)
        store.record_batch(receipt.batch_id, name, "complete" if receipt.ok else "failed")
        if not receipt.ok:
            # A failed batch is a real outcome, not an exception: the node returns
            # a produces-typed artifact so the SOP keeps its type contract, and
            # downstream/objective reflects the failure. (Mirrors loamlite: a wait
            # or failure is a finding, never a crash.)
            log.warning("node %s batch %s not ok: %s", name, receipt.batch_id, receipt.error)
        return parse(receipt, inp, store)

    return Node(name=name, consumes=consumes, produces=produces, fn=fn,
                module=module, description=description)
