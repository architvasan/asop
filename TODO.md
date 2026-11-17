# asop — roadmap

Build order is deliberate: get the **M1→M4 design loop** real and stable first,
then the harder **M5 (reconnaissance) + M7 (mechanism)**, and only then
**M6 (optimization)**, which depends on everything upstream being trustworthy.

Each "write real scientific scripts" task means, concretely, for that module:
- a real **batch script** obeying the `asop.batch.BatchRunner` contract
  (`python <script> input.jsonl output.jsonl`, one row per GPU tile), calling the
  relevant **loamlite skill** runner;
- an **`asop.registry.Adapter`** (`build_rows` store-query + `parse` → typed
  artifact) so an engineer only writes the script;
- a **baseline `BranchRecord`** committed to the registry;
- wiring to `asop.batch.McpBatchRunner` so it runs on the loamlite-managed HPC
  deployment (Aurora/Polaris), replacing `LocalBatchRunner` with no other change.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · *(infra ready)* = scaffold
exists, only the scientific body is missing.

---

## Phase 1 — M1→M4 design loop (do this first)

### 1. Write real scientific scripts for M3 and M4  *(infra ready)*
- [ ] **M3 design** adapter + baseline branches → `DesignBatch`
  - [ ] `rfdiffusion3` backbone generation batch script
  - [ ] `proteinmpnn` inverse-folding (sequence) batch script
  - [ ] (optional 2nd branch) `esm`/PLM seed diversification
  - [ ] register a baseline M3 branch per step (design, sequence)
- [~] **M4 QC** adapter + baseline branches → `QCReport` with a real `objective`
  - [x] `chai` holo co-fold + **ipSAE/ipTM** interface scoring — self-contained
        worker `asop/scripts/m4_qc_chai.py` (vendored numpy-only ipSAE) + adapter
        `asop/adapters/m4_qc.py` (`chai-ipsae` baseline branch); offline mock path
        + tests `tests/test_m4_qc.py`; demo `examples/demo_m4_qc.py`
  - [x] `batch.py` rewritten to loamlite's real **per-row** `run_batch` contract
        (`python step.py input.json` → `result.json`); `McpBatchRunner` for Aurora
  - [x] `pyrosetta` interface ΔG / SC — 2nd M4 branch: worker
        `asop/scripts/m4_qc_pyrosetta.py` (chai co-fold → inline CIF→PDB →
        `rosetta_tool.py interface A_B`, repacked) + adapter
        `asop/adapters/m4_qc_energy.py` (`pyrosetta-dG`); shared build/parse glue
        in `asop/adapters/_common.py`; fork demo `examples/demo_m4_fork.py`
  - [ ] `developability` sequence-level filter (liabilities, charge, aggregation)
  - [x] campaign **objective** = survivor fraction under each branch's gate
        (`objective_definition` reported by the worker)
  - [ ] **objective soundness (blocks trusting the QC fork choice):** ranking two
        *evaluator* branches by their own survivor fraction selects for the looser
        gate, not the better metric (see caveat in `m4_qc_energy.py`). Bind a
        GROUND signal into `QCReport.objective` — agreement with a held-out
        reference or downstream MD/lab confirmation — before letting the operator
        choose a QC method by yield; until then treat orthogonal filters (ipSAE
        AND ΔG AND developability) as sequential funnel steps
  - [ ] weight survivors by developability / node-hour once that adapter lands
  - [ ] validate the real (non-mock) branches on Aurora via `McpBatchRunner`:
        confirm `rosetta_tool.py interface` output keys and the CIF→PDB records
        against a live complex

### 2. Test DAGs with M3/M4
- [ ] `McpBatchRunner` smoke test against a real loamlite HPC deployment
      (one script + JSONL `run_batch`, poll to `complete && ok`)
- [ ] pytest: SOP type-chaining, fork I/O compatibility, stretch-gate
      non-regression, registry commit/reject + reload-from-disk
- [ ] a 1-target M3→M4 run on HPC: designs land in the store, metrics harvested,
      `QCReport.objective` populated, batch receipts recorded
- [ ] stretch-test a real alternative M4 branch (e.g. chai vs boltz) and confirm
      the gate accepts/rejects on real numbers

### 3. Run a design campaign end-to-end M1 → M2 → M3 → M4
- [ ] **M1** knowledge: `fetch-literature` → `rag-build-index` → seed the campaign
      goal / hotspots (feeds loamlite's goal + the module bus)
- [ ] **M2** hypothesis: loamlite Jnana tournament produces target/epitope
      hypotheses; top-k become `TargetSpec`s entering the SOP
- [ ] operator drives the registry-assembled M3→M4 SOP each round
- [ ] `BanditOperator` vs `RandomOperator` A/B on the real forks — first answer to
      "did the agents actually help?"
- [ ] checkpoint/resume across walltime; one human promotion reviewed via the
      `PromotionResult` before a branch goes live
- [ ] **Milestone: a stable M1→M4 campaign on the ALKBH1 target.**

---

## Phase 2 — M5 reconnaissance + M7 mechanism

### 4. Write real M5 and M7 scientific code  *(infra ready)*
- [ ] **M5 recon** adapter → `ReconEnsemble`
  - [ ] `chai` holo fold with PTMs / cofactors (Fe²⁺ + 2OG for ALKBH1)
  - [ ] `bioemu` conformational ensemble
  - [ ] `molecular-simulations` short-MD swarms + MSM; emit `recommended_targets`
        (new `TargetSpec`s) back into the design loop
- [ ] **M7 mechanism** adapter → `MechInsight`
  - [ ] `deepdrivewe-koff` / MSM kinetics for residence time, k_off
  - [ ] free-energy (MM-GBSA → US/metadynamics → alchemical) on top leads
  - [ ] (optional) CALVADOS for biological-context implications

### 5. Test DAGs for M5 and M7
- [ ] M5 recon run: ensemble + recommended targets feed back as `TargetSpec`s
- [ ] M7 run on a top-N shortlist from M4; `MechInsight.changes_scoring` path works
- [ ] bus edges exercised: `M5 → M3/M4` (new structures), `M7 → M2` backward
      (rescore-by-kinetics) actually reweights M4's objective

### 6. Run a design campaign end-to-end using M1–M5 + M7
- [ ] closed loop: M5 recon seeds targets → M3→M4 design/QC → M7 mechanism
      re-ranks leads and feeds insight back to M2/M4
- [ ] **Milestone: kinetics-aware campaign (not just fold confidence).**

---

## Phase 3 — M6 optimization (hardest; last)

### 7. Write real scientific code for M6  *(infra ready)*
- [ ] preference-splitting: mine `PreferenceSet` from (`DesignBatch`, `QCReport`)
- [ ] `diversity-dpo-mpnn` and `negative-design-selectivity-dpo` training scripts
- [ ] GRPO / KTO variants → `ModelDelta` (pointer to updated generator weights)

### 8. Test DAG for M6
- [ ] M6 run produces a `ModelDelta`; M3 picks up the new weights on the next round
- [ ] guardrails: reward-hacking / collapse checks before a `ModelDelta` is trusted
- [ ] stretch-test an optimized generator branch vs the base generator at M3

### 9. Run a design campaign end-to-end M1 → … → M7 (all modules)
- [ ] full loop with the optimization feedback (M4 → M6 → M3) closed
- [ ] **Milestone: self-improving campaign; measure lift vs the Phase-1 baseline.**

---

## Cross-cutting (as needed, not blocking)
- [ ] `LLMOperator.decide` wired to a real model (choice-only prompt); operator
      model size is a cost/latency choice to sweep, not a fixed assumption
- [ ] persist SOP versions + operator history under `state/`; promotion CLI
- [ ] `ResourceManager` backed by loamlite `pool_status` so PE sandboxes never
      starve production
- [ ] bind `QCReport.objective` into loamlite steering so evidence and the DAG
      objective agree
