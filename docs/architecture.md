# asop — architecture

Three views of the `asop` control layer (the SOP-DAG that sits on top of
`loamlite`): the **module/dependency layout**, the **runtime execution flow**,
and the **typed-artifact pipeline** (M1→M7 target state).

GitHub renders the Mermaid blocks below inline. To edit, paste a block into
<https://mermaid.live>.

---

## 1. Module layout & dependencies

```mermaid
graph TD
    subgraph SCHEMAS["schemas.py — typed artifact contracts"]
        ART["Artifact (extra='forbid', origin)<br/>TargetSpec · Design · DesignBatch<br/>Metric · QCReport · ReconEnsemble<br/>PreferenceSet · ModelDelta · MechInsight<br/>ForkChoice · BranchProposal · PromotionResult"]
    end

    subgraph CORE["dag.py — the SOP-DAG (plain-Python orchestrator)"]
        NODE["Node (frozen): consumes→produces, fn"]
        FORK["Fork: interchangeable branches<br/>allow_operator gate"]
        SOP["SOP (frozen, versioned)<br/>type-chain checked · .sub() · .with_branch()"]
        EXEC["Executor.run(sop, initial, ctx)<br/>→ (artifact, trace)"]
        SEL["Selector = (ctx, fork) → (branch, by, why)"]
    end

    subgraph CONTROL["control layer"]
        OP["operator.py<br/>BanditOperator · LLMOperator · RandomOperator"]
        ENG["engineer.py<br/>ProcessEngineer: smoke + stretch gate"]
        REG["registry.py<br/>Registry · BranchRecord · Adapter<br/>append-only JSONL · build_sop()"]
    end

    subgraph INFRA["infrastructure"]
        MOD["modules.py<br/>M1–M7 REGISTRY · ResourceManager"]
        BUS["bus.py<br/>Bus · ALLOWED edges (incl. M7→M2)"]
        STORE["store.py<br/>SQLite Store (designs/metrics/batches)"]
        BATCH["batch.py<br/>BatchRunner · LocalBatchRunner · McpBatchRunner"]
        NODES["nodes.py<br/>batch_node() factory"]
    end

    subgraph SCI["M4-QC scientific code (item 1)"]
        COMMON["adapters/_common.py<br/>build_rows + metric-agnostic parse"]
        AQC["adapters/m4_qc.py<br/>ADAPTER 'm4-qc' · chai-ipsae baseline"]
        AEN["adapters/m4_qc_energy.py<br/>ADAPTER 'm4-qc-energy' · pyrosetta-dG"]
        WCHAI["scripts/m4_qc_chai.py<br/>Chai co-fold + vendored ipSAE"]
        WROS["scripts/m4_qc_pyrosetta.py<br/>Chai + CIF→PDB + interface ΔG"]
    end

    CORE --> SCHEMAS
    OP -->|provides select| SEL
    SEL -.-> EXEC
    ENG -->|PromotionResult| REG
    REG -->|build_sop| SOP
    REG --> NODE
    EXEC --> SOP

    NODES --> BATCH
    NODES --> STORE
    AQC --> COMMON
    AEN --> COMMON
    COMMON --> NODES
    AQC -->|make_node| NODE
    AEN -->|make_node| NODE
    AQC -.frozen script.-> WCHAI
    AEN -.frozen script.-> WROS
    REG -->|Adapter| AQC
    REG -->|Adapter| AEN

    EXEC --> MOD
    EXEC --> BUS
```

---

## 2. Runtime execution flow (one SOP round)

```mermaid
graph TD
    PE["Process-engineer<br/>authors real batch script (uses skills)"] -->|smoke + stretch gate| GATE{"non-regressing<br/>on held-out probes?"}
    GATE -->|no| REJECT["rejected — not committed"]
    GATE -->|yes, human-gated| REGc["Registry.register(BranchRecord)<br/>append-only commit"]

    REGc --> BUILD["Registry.build_sop(version, step_order)"]
    BUILD --> SOPi["immutable SOP<br/>(Fork per step)"]

    SOPi --> EX["Executor.run(sop, initial, ctx)"]
    EX --> STEP{"step type?"}
    STEP -->|Node| RUN1["node.run → fn"]
    STEP -->|Fork & allow_operator| OPsel["operator.select(ctx, fork)<br/>ε-greedy / LLM / random"]
    STEP -->|Fork closed| DEF["default_branch (deterministic)"]
    OPsel -->|branch NAME only| RUN2["branch.run → fn"]
    DEF --> RUN2

    RUN1 --> NF["batch_node fn"]
    RUN2 --> NF
    NF --> GET["get_store / get_runner from ctx.extra"]
    GET --> ROWS["adapter build_rows: query Store → JSONL rows<br/>(+ holo cofactor ligands)"]
    ROWS --> SUB["runner.submit(step, script, rows)"]

    SUB --> PERROW["PER-ROW: python script.py input.json → result.json<br/>(one row per GPU tile on Aurora)"]
    PERROW --> HARVEST["harvest result.json per row"]
    HARVEST --> PARSE["adapter parse: write Metric rows → Store,<br/>survivors by gate, objective + objective_definition"]
    PARSE --> QC["QCReport artifact"]
    QC --> REC["operator.record(fork, branch, objective)<br/>→ updates BranchStats"]
    REC -.->|feeds next round's ε-greedy| OPsel
```

---

## 3. Typed-artifact pipeline (M1→M7 target state)

Green = built (M4 fork + both branches) · amber = partially scaffolded
(M3 adapter is next) · grey = infra-ready, scientific body not yet written.
Matches `TODO.md`.

```mermaid
graph LR
    M1["M1 knowledge<br/>literature / RAG"] -->|goal, hotspots| M2["M2 hypothesis<br/>Jnana tournament"]
    M2 -->|TargetSpec| M3["M3 design<br/>rfdiffusion → proteinmpnn"]
    M3 -->|DesignBatch| M4["M4 QC (fork)<br/>chai-ipsae | pyrosetta-dG"]
    M4 -->|QCReport| M6["M6 optimization<br/>DPO / ModelDelta"]
    M6 -.ModelDelta.-> M3
    M5["M5 recon<br/>ensemble / MD"] -->|ReconEnsemble →<br/>recommended TargetSpecs| M3
    M4 -->|top-N| M7["M7 mechanism<br/>kinetics / free energy"]
    M7 -.MechInsight (rescore).-> M2

    classDef done fill:#bbf7d0,stroke:#16a34a,color:#064e3b;
    classDef wip fill:#fde68a,stroke:#d97706,color:#713f12;
    classDef todo fill:#e5e7eb,stroke:#9ca3af,color:#374151;
    class M4 done
    class M3 wip
    class M1,M2,M5,M6,M7 todo
```

---

## Reading notes

- **The operator's only influence on control flow is the branch _name_**
  (view 2, `OPsel → branch NAME only`). Everything else is deterministic Python
  — the closed-choice contract.
- **The registry is the single trust boundary**: a script only enters the
  running SOP through `register()`, gated by the stretch test and a type-check of
  the adapter's I/O.
- **`_common.py` is the shared seam** both QC adapters sit on, so a third QC
  branch needs no new glue.
