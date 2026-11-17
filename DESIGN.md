# asop — design notes

This file reconciles the inconsistencies in `README.md` and documents the code
scaffold under `asop/`. asop is a **control layer on top of
[`loamlite`](../matts_crazy_idea)** — it does not re-implement agents, execution,
skills, or the Jnana loop.

## What asop reuses from loamlite vs. what it adds

| Concern | Owner |
| --- | --- |
| Hypothesis generation + Elo ranking (M2) | **loamlite** (Jnana tournament) |
| Validator / code-generator / reviewer / reasoner agents, typed outputs | **loamlite** |
| HPC execution (Globus→PBS→Academy→Ensemble Launcher), `run_batch` | **loamlite** |
| Scientific skills (chai, rfdiffusion3, proteinmpnn, md-simulation, rag-*, ...) | **loamlite** |
| Literature/RAG knowledge (M1) | **loamlite** skills |
| **SOP-DAG** of typed steps (the M3→M6 production loop) | **asop** (`dag.py`) |
| **Operator** — selects a branch at an *allowed* fork | **asop** (`operator.py`) |
| **Process-engineer** meta-loop — propose/smoke/stretch/promote | **asop** (`engineer.py`) |
| **DAG registry** — PEs commit validated real code; SOP assembled from it | **asop** (`registry.py`) |
| **Deterministic node bodies** — store-query → `run_batch` → harvest | **asop** (`nodes.py`, `batch.py`) |
| **Record store** the nodes read/write (SQLite) | **asop** (`store.py`) |
| **Module registry + node arbiter** | **asop** (`modules.py`) |
| **Typed module bus** (forward/backward comms) | **asop** (`bus.py`) |
| Artifact data contracts on DAG edges | **asop** (`schemas.py`) |

## Canonical module numbering (fixes README off-by-ones)

The README's "details" sections are shifted one module down from the modules
they name. The single source of truth is `modules.REGISTRY`:

| Code | Module | Process engineers (nodes) |
| --- | --- | --- |
| M1 | Knowledge gathering | — |
| M2 | Hypothesis generation (= loamlite) | — |
| M3 | Binder design | design, sequence (2) |
| M4 | **Quality control / evaluation** | evaluation (2) |
| M5 | **Target reconnaissance** (structure+MD+MSM) | structure, md-sim, md-analysis (6) |
| M6 | **Optimization** (DPO/KTO/GRPO, preference splitting) | preference, optimizer (1) |
| M7 | Mechanistic understanding | — |

Key corrections vs. README prose:
- target structure prediction / MD = **M5**, not M4;
- optimization & "optimized weights → M3" = **M6**, not M5;
- "Five modules" → **seven**.

## Operators select real code, not skills (the core division of labor)

The production loop never depends on skill-following or code-generation at run
time. That keeps it fast and reproducible, and it decouples the model that runs
production from the model sizes used elsewhere — the operator's job is a bounded
decision, so it can be served by whatever model fits the budget. Hence:

- **A DAG node body is a real, deterministic Python function** (`nodes.batch_node`)
  that issues a deterministic query against the record **store** (`store.py`) and
  fans the work across HPC with one `run_batch` (`batch.py`, matching loamlite's
  one-script+JSONL contract). No LLM inside a node.
- **The operator only *names a branch*.** It is handed, per branch, a name, a
  one-line description, and a running mean objective — and returns one name
  (`operator.LLMOperator`, with `BanditOperator` as both default policy and
  fallback). Branch selection is a closed multiple-choice.
- **Skills live with the code-authoring roles** — the process-engineers (who
  author node code) and the hypothesis scientists. A skill-authored function is
  not trusted until it is frozen into deterministic code and passes the gate.

## The DAG registry (`registry.py`)

Process-engineers **write real code and commit it to an append-only registry**
once it is proven valid. The registry is the single source of truth for what the
production DAG is:

```
  process-engineer (uses skills to author code)
     writes a real batch SCRIPT ──► engineer.evaluate (smoke + stretch)
     proven non-regressing?      ──► registry.commit_promotion(...)   [commit]
                                        │
  operator (closed-choice) ──► Registry.build_sop()  assembles the SOP from every
     picks a branch NAME                registered branch, and runs it.
```

What is stored is serializable: the branch's **script** (the real code) + an
`Adapter` name (module-standard store-query/parse glue) + provenance
(who proposed it, the objectives at promotion, the version). Append-only +
versioned ⇒ any past DAG is reconstructable and rollback is trivial. The registry
validates on write (unknown adapter, duplicate branch, I/O-type mismatch, and
refuses to commit a rejected proposal), so it is the trust boundary.

Demonstrated offline end-to-end by `examples/demo_registry.py` and
`examples/demo_batch_node.py` (real subprocess batches, SQLite store, no LLM/HPC).

## The SOP-DAG (`dag.py`)

- A `SOP` is an **immutable, versioned** tuple of `Step`s. A step is a `Node`
  (one implementation) or a `Fork` (interchangeable branches).
- Every step declares `consumes`/`produces` **artifact types**. `SOP.__init__`
  checks the pipeline chains; `Fork` checks all branches share its I/O types.
  **This type contract is the safety rail that makes agentic branch selection
  safe** — an operator or a promoted branch can never emit something a
  downstream step cannot read.
- The only run-time decision is *which branch* at a fork whose `allow_operator`
  a human has opened. Everything else is deterministic Python (loamlite's
  philosophy, one level up).
- Promotion never mutates a running SOP: `SOP.with_branch(...)` returns a new
  SOP at a new version → free rollback + audit trail.

## Separation of powers (safety)

- **Operator** (`operator.py`) may only *name* a branch at an allowed fork. It
  cannot add/edit/remove branches or touch other steps. Default policy is an
  explicit, auditable ε-greedy bandit over each branch's historical campaign
  **objective** (the README had no objective for the operator; this supplies
  one). An LLM operator plugs into the same `Selector` signature.
- **Process-engineer** (`engineer.py`) may only *propose* branches and must pass
  a gate before promotion; it never selects branches in production.
- **Promotion** (opening a fork to the operator) is human-gated: `evaluate`
  returns a candidate SOP, it does not deploy it.

## The promotion gate (makes README's "strict constraint test" real)

The gate is deliberately **minimal**. Its job is to certify a change is **safe to
try in production** — it does not break the DAG, does not regress the current
objective, and is affordable — **not** to prove the change is an improvement.
Proving improvement is usually impossible before production and is the PE's and the
analysis scientist's job, not the gate's; over-building the gate would both
duplicate the operator's production A/B and encroach on the roles it exists to
protect. A proposed branch is accepted only if:
1. **smoke** — runs on one probe and produces the fork's declared output type;
2. **stretch** — the README's own rule `max(i-2, 0) → end`: splice the branch in,
   run the downstream stretch on a **held-out probe set** to completion, and
   require the terminal **objective** to be **non-regressing** vs. the current
   production branch on the same probes (± `epsilon`);
3. **promotion** — emit a new SOP version with the fork opened to the operator.
4. **speed** — a metric proposed as a gate must also pass the cost/throughput test
   (see "A gate metric must be cheap and sound"); a sharp-but-slow metric is
   relegated to a validation anchor, not a gate.

**Where "better" is actually decided.** The gate only opens a *safe* branch to the
operator. The operator's bandit then accumulates that branch's **real objective**
over production rounds and converges on what genuinely wins, and the **M7 analysis
scientist** interprets the raw outcomes to judge whether an improvement is real and
why. Improvement is therefore an empirical, post-hoc, production verdict — the
README's "either the agents improve the DAGs, or we at least have data." The gate
is the floor; production + M7 is the proof.

## Objective

Everything asop optimizes is the scalar `QCReport.objective` (M4), with a
human-readable `objective_definition`. Make this the real campaign goal
(e.g. *developable, selective, high-confidence binders per node-hour*) so the
README's honest "did the agents actually help?" question is measurable: compare
`BanditOperator` vs. `RandomOperator` on the same forks.

The whole PE/operator loop is only as trustworthy as that objective, so the
objective must be both **sound** (hard to game, generalizable across modules) and
**cheap** (runnable in the hot loop). The next two sections make each concrete.

## Objective soundness — what makes a gate metric trustworthy

**Scope — don't over-build this.** For an ordinary branch proposal (a new design or
QC method) the minimal gate above is enough: safe to try, then let production
decide. The machinery below applies to one narrower, higher-stakes case — when a PE
proposes a metric to serve **as the objective/gate that other PEs are measured
against** (a "gold standard"). A measuring stick must itself be trustworthy, so that
case — and only that case — earns the extra checks.

### The trap
Ranking two *evaluator* branches by their own survivor fraction selects the
**looser gate**, not the better metric — the agent "improves" the number while
producing nothing better (Goodhart). Any threshold-based self-report has this flaw.

### Why not domain-specific negative controls
They do not generalize. Sequence **scrambles** are only valid negatives when
residue order drives structure — they fail for short peptides, where a scramble ≈
the real design. **Hotspot knockout** assumes an identifiable binding interface,
which is meaningless for e.g. enzyme design. So a metric is judged purely from how
its values are **distributed across a population of designs** — no controls, no
labels, no task-specific assumptions:

1. **Reliability floor — discrimination index (ICC).** Run the metric with its
   natural stochasticity (folding seeds, stochastic packing, MD replicas) to get
   several values *per design*, then
   `ICC = (MS_between − MS_within) / (MS_between + (k−1)·MS_within)` ∈ [0,1]: the
   fraction of the metric's variance that is real design-to-design signal rather
   than measurement noise. Rejects a **degenerate** metric (all designs at the
   ceiling → ICC≈0) and a **noisy** one (seed scatter swamps signal → ICC≈0).
   Threshold-free; needs only replicates, which every stochastic scorer produces.
   (This is the statistical core of the HTS **Z′-factor**, minus Z′'s requirement
   for defined positive/negative control populations — which is what made Z′
   non-generalizable.)
2. **Validity — orthogonal inter-metric agreement.** Spearman correlation across
   the population between two *independent* estimators of the same quantity
   (ML folding confidence vs physics energy; two folders; an ML activity predictor
   vs a physics barrier for enzymes). Every module has ≥2 ways to score. High ICC
   but low agreement = the metric reliably measures the WRONG axis — caught here.
3. **Promotion — one-sided distributional non-regression.** The stretch gate
   compares the new metric's *distribution* to the baseline's on held-out designs
   and requires it is not stochastically worse (one-sided; on the relevant
   quantile — the top-k tail for design yield, since design cares about the tail).

### Reliability ≠ validity
Distribution-only criteria measure resolving power, not correctness: a metric can
have ICC≈0.9 and resolve designs along an irrelevant axis. ICC + orthogonal
agreement is the generalizable floor; the ultimate grounding is correlation with a
small, expensive **gold set** (MD stability / wet-lab), run rarely as an anchor
(see cost below).

### Grounding chains up the pipeline
A metric is never trusted in the abstract. **M4 earns trust by passing the above,
and M3 is then judged by "yield of designs that survive validated-M4"** — not by a
raw folding proxy such as ipTM, which is itself gameable. Trust flows upward so
each layer's objective is grounded in a downstream layer that already passed its
own validity test.

### Measurability
This keeps the README's honest question answerable: once the objective is sound,
compare `BanditOperator` vs. `RandomOperator` on the same forks to see whether
opening a fork to the operator actually beats the random baseline.

## A gate metric must be cheap and sound (the speed test)

A gate/objective runs in the **hot loop** — on every design, every round, inside
PE evaluation and operator selection. A metric that is sharp but expensive
(e.g. 3 h/design) is non-functional *for gating*, however good it is. So:

- **Role decides the budget.** A metric used **as a gate** must be cheap; a sharp,
  expensive metric is not banned — it is relegated to a **validation anchor**, run
  rarely on a small held-out set to validate (correlate with) the cheap gate.
  Cost per design should scale inversely with how many designs the metric sees:
  tiny at M3/M4 (sees everything), large at M7 (sees only a top-N shortlist). This
  is the tiered-funnel principle — cheap filters first, expensive metrics only on
  the survivors.
- **Speed test at promotion.** When a PE proposes a metric as the gold standard
  other PEs gate against, it must pass a throughput test *before* promotion:
  measured **node-hours (GPU-tile-seconds) per design**, taken for free by timing
  the **smoke run** at steady-state `run_batch` throughput (not cold-start
  latency), projected as `cost/design × designs/round` and checked against the
  step's allocation in `modules.ResourceManager`. Over budget ⇒ fails the gate
  role (may still register as an anchor).
- **Cost enters the ranking, not just a floor.** The campaign objective is quality
  **per node-hour**: among adequately-sharp metrics, prefer the cheaper — a metric
  90% as sharp at 1% of the cost is the better gate (the same cost/latency logic by
  which the operator's model size is chosen).
- **Replicate cost counts.** The ICC criterion needs replicates, which multiply
  cost: a metric needing 20 seeds to reach ICC>0.5 is ~20× more expensive than one
  reliable in 1–2 seeds. The right figure of merit is **reliability per node-hour**
  — the cost projection must include the replicates needed to reach an adequate ICC,
  which unifies the sharpness and speed constraints into one axis.

## Module bus (`bus.py`)

Forward (knowledge/ideas → engineers) and backward (requests/findings →
scientists) edges are **declared** in `ALLOWED` and checked on `send`. The
README's one contradiction — M7 has "no backward need" yet M7 insight "transforms
the hypothesis scientist's understanding" — is resolved by **keeping the
`M7→M2` backward edge** (mechanistic insight that reshapes hypotheses / rescoring).

## Status

- Implemented and exercised offline (deterministic, no HPC/LLM):
  - `dag.py`, `operator.py`, `engineer.py`, `modules.py`, `bus.py`, `schemas.py`
    → `examples/demo_sop.py`
  - `store.py`, `batch.py` (`LocalBatchRunner`), `nodes.batch_node` → real
    subprocess batches reading a SQLite store → `examples/demo_batch_node.py`
  - `registry.py` → PE-writes-code → stretch gate → commit → SOP re-assembly →
    operator selection → `examples/demo_registry.py`
- **Documented but not live:** `batch.McpBatchRunner` (direct client to
  loamlite's hpc-deployment MCP server — real `run_batch` on Aurora/Polaris);
  the `LLMOperator.decide` callback (wire to a vLLM/pydantic-ai agent with a
  `Literal[...]` output); M3/M5/M6 adapters; binding `QCReport.objective` into
  loamlite's steering.

## Next steps (suggested order)

1. Write the real M4-QC script (chai/boltz ipSAE) and an `McpBatchRunner` wired to
   the loamlite deployment server, so `demo_registry`'s stub script is replaced by
   production code with no change to the surrounding machinery.
2. Add the M3 adapter (rfdiffusion3 → proteinmpnn) emitting `DesignBatch`, and the
   M5 recon adapter (`chai` holo + `bioemu` + MSM) emitting `ReconEnsemble` whose
   `recommended_targets` feed back as new `TargetSpec`s — closing the design loop.
3. Back `LLMOperator.decide` with a real model and A/B it against `RandomOperator`
   on the same forks to answer the README's "did the agents help?" (operator model
   size is a cost/latency choice to sweep, not a fixed assumption).
4. Add a human promotion CLI over `registry.commit_promotion` (review the
   `PromotionResult` + diff before the branch goes live).
