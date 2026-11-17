# asop
##Agentic Standard Operating Procedure

### Binder Design System

#### Five modules
- M1: Knowledge Gathering
    - Web lookup
    - Paper extraction/compilation
    - Vector database compilation
    - Knowledge synthesizing

- M2: Hypothesis Generation
    - RAG 
    - LLM hypothesis generation
    - Jnana tournament
    - Freeform validation methods

- M3: Binder design
    - Design methods
        - Diffusion models
        - Protein language models
        - Inverse folding
        - Structure prediction 

- M4: Quality control 
        - Sequence based evaluations
        - Structure prediction 
        - Short simulation
        - Biophysical measurements
        - ADMET models

- M5: Target reconnaisance
    - Target structure prediction
        - With PTMs
        - Cofolded with interacting partners
        - BioEMU ensemble prediction
    - MD simulation
        - Large scale short simulations
        - CVAE based collective variable
        - MSM constructions
    - MD analysis methods

- M6: Optimization
    - DPO
    - KTO
    - GRPO

- M7: Mechanistic understanding
    - Long binder-protein simulations
    - Deepdrive WE binder->protein binding pathway
    - CALVADOS simulations to understand binder biological implications
    - Free energy simulations

#### SOP system 
##### DAG
- M3-5: always running on set resources with a deterministic algorithm (SOP) built from a DAG
- DAG is predefined by a human initially
- Initially DAG is ->->-> (no branching)
- Process engineers are allowed to add branches to the DAG once a branch has been sufficiently validated
- An "operator" agent is allowed only to pick what the next branch should be in a fork
- If the fork is not allowed then a branch is chosen randomly

##### Process engineers
Each M3-5 has a "process engineer" who proposes new process ideas and tests them on a small number of resources
    - Once an idea passes a strict constraint test (validation/real smoke test) -> idea is written as a function within the DAG
    - A stretch of the DAG  (max {(proposedstep - 2), 0} -> end) is tested to ensure the newly proposed idea actually works
    - Once an idea is validated it is added as a branching point within the DAG an "operator" agent is allowed to choose that branching point
###### M3 details
- A process engineer each for binder design and evaluation
- Each engineer gets 2 nodes to play with
###### M4 details
- Target structure prediction engineer (1 node)
- MD simulation engineer (4 nodes)
- MD analysis engineer (1 node)
###### M5 details
- Preference splitting engineer
- Optimization engineer (1 node)

##### Agentic communication networks
- M1 knowledge librarian <-> M2 hypothesis scientists
                          <-> M3 process engineers
                          <-> M4 process engineers
                          <-> M5 process engineers
                          <-> M6 process engineers
                          <-> M7 mechanistic scientists
    * Forwards direction: providing necessary knowledge to inform hypotheses/mechanism investigation/engineering 
    * Backwards direction: request for further knowledge from the scientists and engineers

- M2 hypothesis scientists <-> M3 process eng
                           <-> M4 process eng
                           <-> M5 process eng
                           <-> M6 process eng
                           <-> M7 mechanistic sci
    * Forwards direction (not M6) : providing potential breakthrough ideas for engineers to try to implement
    * Forwards (M7): context to help M6 investigate mechanisms
    * Backwards (not M7): was that idea actually useful or not?
    * Backwards (M7): mechanistic/analytical discovery which can transform hypoth scientist's understanding

- M7 mechanistic scientist -> M3 PE
                           -> M4 PE
                           -> M5 PE
                           -> M6 PE
    * Forwardsd direction: understanding that certain types of binders are mechanistically useful (beyond just simple folding confidence/mmpbsa)
    * Backwards direction: not needed since the PEs care about efficiency/effectivenss and the scientist cares about understanding. Efficient systems aren't inherently helpful for improving understanding
##### Deterministic communication networks
- M1 knowledge database -> M2 RAG
- M3 binder designs (sequence + structures) -> M4 quality control evaluations
- M5 target reconnaisance structures -> M5 reconnaisance analysis -> M3 + M4 process engineers (should they add those new structures to the DAG for developing binders?)
- M3 binder seqs + structures + M4 evaluations -> M6 optimization preference splitting
- M5 optimized model weights -> M3 design models
- M3 designs (seqs + structures) + M4 evaluations -> M7 mechanistic scientist
##### Catalogue of agents
- M1 knowledge librarian: curate and collect the right information for other agents
- M2 hypothesis scientist: take in knowledge and output creative ideas which may be potential breakthroughs
    - Open ended exploration is allowed and highly encouraged (they are within their sandboxes and can play around)
    - Analogous to principal investigators/professors in a lab
    - Some of their ideas may be cuckoo for cocoa puffs but we want educated creativity with few limits 
- M3-6 process engineers: take in knowledge + hypotheses and try out new processes to improve their specific step in the pipeline
    - Only focused on that aspect of the pipeline... not scientists, they care about efficiency and effectiveness
-M7 mechanism scientist: take the mechanistic outputs of M3-6 as well as M1 knowledge and M2 ideas and try to contextualize the results to form real grounded conclusions
    - Analogous to the graduate students
    - Actually go through raw data and understand how things fit together
    - Allowed open ended exploration using free tools but things need to tie to the output data of M3-6.
    - Open ended exploration of data and the hypoth scis ideas are useful but it is not tied to them

##### Initial setup required
- venv crap (academy, globus compute, mcp, pydantic ai)
- starting, working DAGs for M3-6
- goals for all modules

##### End result
- Either agents improve the outcomes of the DAGS or
- Agents are stupid and at least we have data
