# Research-to-Implementation Plan: Biofilm, Quorum Sensing & Multiphysics Solver Coupling

**Date:** 2026-10-30  
**Status:** proposed research branch plan; no external solver has been modified.

## Executive decision

Advance the existing Julia-first flow-cytometry-model-bridge through explicit, conservative coupling interfaces to OpenFOAM and PHREEQC. Do not embed biological signaling, cell-state transitions, flow-cytometry likelihoods, or nanocarrier classification directly into either solver. Keep Julia as the biological/regulatory/constitutive orchestration layer, OpenFOAM as the spatial transport/hydrodynamic solver, PHREEQC as the aqueous speciation/mineral-equilibrium engine, and flow cytometry as an observation operator.

Begin with offline, one-way snapshot coupling and conservation/unit audits. Introduce two-way feedback only after exchange variables, units, timing, compartment definitions, boundary conditions, and mass/charge balances are verified.

This plan uses the October 23–30 and related October 15–19 research summaries supplied for this project. The findings below are experimental motivations and candidate model extensions; they do not provide universal kinetic constants or validate a unified model.

## Current repository baseline

The main branch currently contains:
- `code/FlowCytometryObservationModel.jl`, with explicit volume conventions, count/gate likelihoods, synthetic observation simulation and recovery, calibration, uncertainty, identifiability/equifinality, held-out checks, and OpenFOAM/PHREEQC snapshot contracts.
- `code/MicrofluidicClassificationEngine.jl`, with FCS ingestion, cytometry processing/gating, a synthetic-data-trained classifier, microfluidic trajectory joining, and shared domain-separated visualization physics.
- `solve_core_multishell_rd` for a predicted Ghosh-track radial core–multishell reaction–diffusion model.
- `solve_emulsion_interfacial_droplet` for a distinct Iain Jack emulsion-interface model.
- Browser workbench at `workbench/microfluidic_cytometry_workbench.html`.

The supplied repository manifest records 118/118 observation-model assertions and 621/621 classification-engine assertions passing under Julia 1.13.1, with cross-language reference verification. It explicitly records that external OpenFOAM/PHREEQC production integration has not yet occurred.

**Test interpretation:** the existing suites establish internal implementation behavior for the tested cases. The classification suite uses simulated events and synthetic labels; its high scores do not establish empirical classification performance on Ghosh's experimental samples. Likewise the shell and emulsion simulations use prescribed illustrative parameters and do not establish measured physical behavior.

## Solver responsibilities

| Component | Owns | Exports to coupling layer |
|---|---|---|
| OpenFOAM | Flow/pressure, advection, fluid shear, residence time, spatial transport, and any explicitly implemented fluid/transport mechanics | velocity, pressure, shear, transport fluxes, spatial concentration fields, boundary fluxes and mesh/time metadata |
| PHREEQC | Aqueous speciation, activity corrections, mineral equilibria, selected aqueous/mineral kinetics | pH, pe/Eh assumptions, ionic strength, dissolved species, saturation indices and reaction/source/sink terms |
| Julia | Biological state, cellular nutrient quota, QS/QQ network, stress and phenotype transitions, EV production, EPS/PGA states, constitutive matrix relations, model orchestration, inference and uncertainty | state-dependent rates, matrix property fields, biological source/sink terms, spatially and temporally defined reaction terms |
| Flow cytometry / imaging observer | Measurement and data processing, not physical transport | observed gate counts, calibrated concentrations, classifier outputs, measurement uncertainty and provenance |

Flow cytometry is not an in-situ spatial sensor. It cannot independently identify pore-scale location, local EPS cross-link density or spatial matrix morphology. Pair it with confocal/endpoint microscopy, diffusion measurements, spectroscopy, microscopy or other independent measurements as appropriate.

PHREEQC should not become a generic biological signaling engine. It should provide chemically meaningful aqueous/speciation calculations. QS production, perception, transcriptional regulation and EV biogenesis remain biological model processes in Julia unless a narrowly defined chemical reaction is deliberately represented in PHREEQC.

## Research-driven model extensions

### A. Biofilm matrix swelling and mechanics — PNAS, October 2026

The supplied paper summary describes complementary roles for PGA (water uptake/swelling) and EPS (matrix cross-linking/structural integrity) in a particular B. subtilis system. This motivates an explicitly testable matrix model with distinct state variables, for example:
- local PGA amount or volume fraction;
- local EPS amount or effective cross-link proxy;
- matrix hydration/swelling;
- effective elastic or poroelastic properties;
- mechanical stress/strain and morphology response.

Julia should own the candidate constitutive relationships and biological production/degradation terms. OpenFOAM should receive a deformation/permeability/transport update only when those fields are physically part of the chosen coupled problem. Do not transplant B. subtilis parameters into other organisms or clay/microfluidic systems.

### B. Nutrient-dependent quorum sensing, stress response and EV biogenesis — mBio, October 2026

Represent separately:
- nutrient availability and (if supported) intracellular nutrient quota;
- QS signal abundance and degradation;
- regulatory/stress response;
- species- and strain-specific transcriptional topology;
- EV production and release as downstream outputs.

The supplied S. aureus agr/alpha-PSM/RNAIII findings motivate this architecture but do not supply universal kinetics. Keep the regulatory network species-specific and calibrate against appropriate transcript/protein/EV measurements before predictive claims.

### C. Matrix-dependent heterogeneous diffusion — supplied UPEC study, October 2026

Move beyond a single uniform biofilm diffusivity when data support spatially varying transport:
- D_eff may depend on matrix composition, location, molecular size and charge;
- distinguish diffusion from reversible adsorption/retention and chemical reaction;
- represent boundary conditions and source/sink terms explicitly.

OpenFOAM should solve the spatial transport fields. Julia can provide constitutive closures or parameter fields fitted to experiments. PHREEQC supplies aqueous chemical conditions and relevant speciation where appropriate. The supplied study's curli and peptide findings are strain/matrix/molecule-specific, not general coefficients.

### D. Localized delivery / bioelectronic perturbation — supplied October 28 paper

Represent spatially localized delivery as a source term S_i(x,t) with independently specified units, support, time profile and mass input. OpenFOAM solves transport; Julia describes biological response; PHREEQC participates only where aqueous speciation/reaction is relevant. The paper's surface-attached B. subtilis system is an experimental motivation, not validation of a 3D microfluidic or core–multishell system.

### E. Species-specific quorum-sensing topology — supplied PqsE promoter paper

Model regulatory network topology separately from parameter values. A species-specific missing/added regulatory edge cannot always be represented by merely changing a rate coefficient. Keep alternative topologies explicit and compare them against strain-matched measurements.

### F. Nutrient quota and bioprocess kinetics — supplied Droop-model paper

Represent extracellular substrate and intracellular quota as separate variables where justified. Calibrate uptake, quota dynamics, growth and production against batch data before using continuous-culture or reactor-scale predictions. The yeast bioprocess model's parameter values are not transferable to bacterial biofilms.

## Coupling sequence

### M0 — Freeze contracts before touching solvers

For every exchanged field, specify:
- physical meaning and unit;
- spatial support (cell/face/volume/compartment);
- coordinate system and mesh identifier;
- vessel/chip/sample/compartment identity;
- timestamp, timestep and interpolation rule;
- sign convention and whether a value is concentration, flux, rate, fraction or state;
- epistemic label (`observed`, `calibrated`, `assumed`, `predicted`);
- source hash, solver version, configuration and provenance.

Require exact time/compartment alignment and explicit mapping between OpenFOAM mesh fields and Julia/PHREEQC compartments. Extend current snapshot structs only after defining the exchange schema.

### M1 — Offline one-way snapshot coupling

Export OpenFOAM transport fields and PHREEQC speciation snapshots to a versioned, machine-readable schema. Import to Julia and evaluate the biological model without feeding values back into either solver.

Acceptance:
- units and sign conventions pass tests;
- vessel/chip/time/compartment alignment is enforced;
- input/output hashes and solver versions are recorded;
- no hidden interpolation or extrapolation;
- mass/element balance checks are reported;
- synthetic integration tests and independent hand-calculated cases pass.

### M2 — Spatial transport and heterogeneous diffusivity

Add a bounded spatial diffusivity/source-term interface. Validate first against simple analytical diffusion/advection cases and measured tracer diffusion data. Check mesh/time-step convergence, nonnegative concentration where physically required, and integrated mass flux.

Do not assume that a matrix-dependent D_eff law is automatically a PHREEQC function; it is usually a Julia constitutive closure or OpenFOAM transport property supplied by calibrated fields.

### M3 — Biological and regulatory states

Add the matrix, nutrient quota, signaling, stress and EV modules in separate Julia components. Each mechanism should be independently switchable, with separate parameter sets and priors. Compare simpler and more complex models rather than assuming all proposed mechanisms are active.

Acceptance:
- state variables have explicit units and bounds;
- biological source/sink terms are mass-consistent;
- identifiability/sensitivity analyses are conducted;
- parameters are constrained by appropriate independent measurements;
- competing regulatory topologies and alternative mechanisms are compared.

### M4 — Two-way coupling only where justified

Introduce biological feedback to transport/mechanics only where a mechanistic relationship is specified and measurable, e.g. matrix-dependent porosity/permeability or swelling-induced geometry changes. Use conservative exchanges and a documented staggered or iterative coupling scheme. Demonstrate stability with timestep and coupling-iteration convergence tests.

### M5 — Empirical predictive validation

Use held-out biological replicates, chips, vessels or time points—not random event-level splits. Validate independent observables:
- spatial diffusion maps / tracer penetration;
- matrix composition and hydration;
- morphology and mechanics;
- time-resolved signal/EV measurements;
- cytometry population proportions and absolute counts;
- microfluidic transit trajectories;
- any study-specific functional endpoint.

Report predictive intervals and failure cases. Do not call a model empirically validated based on synthetic-data recovery or internal unit tests alone.

## Track-specific boundaries

### Track 1 — Dr. Ghosh / nanobio and microfluidic classification

Use Julia core–multishell reaction–diffusion and the microfluidic classifier, with OpenFOAM only when the channel/device fluid mechanics materially affect transport. PHREEQC is optional and should be included only if relevant aqueous speciation or mineral chemistry exists in the actual assay.

The spherical shells are predicted concentration fields, not observed physical layers unless validated by imaging. The classifier should be trained and evaluated on expert-labelled, assay-specific experimental data with held-out chips/runs. Distinguish fluorescence association/uptake from demonstrated internalization.

### Track 2 — Clay–microbes–metal pilot

Use OpenFOAM for transport/shear, PHREEQC for aqueous chemistry/mineral equilibria, Julia for biological/matrix dynamics, and flow cytometry as an exploratory secondary observer on the specific sampled compartment. Preserve maximum pit depth per coupon as the primary corrosion endpoint. Require clay-only matrix blanks, counting beads, extraction-recovery validation, controls for staining/disaggregation and sample-origin tracking.

### Track 3 — Iain Jack / bitumen emulsion and interfacial rheology

Keep the existing emulsion droplet/interface model and constitutive fitting separate. Use Julia for temperature-dependent constitutive relations/uncertainty and OpenFOAM for multiphase/interfacial flow as needed. Validate with DSR and moisture-susceptibility evidence. Do not attach biological release shells, flow-cytometry observation models or clay-microbe PHREEQC chemistry to this track. Share rendering, provenance and data-contract infrastructure only.

## Implementation cautions found in current source

1. The synthetic-data generator in `MicrofluidicClassificationEngine.jl` assigns feature distributions and labels by construction. Strong cross-validation performance therefore demonstrates recoverability under the generator, not real-world assay classification. Add experimental validation when real de-identified, expert-labelled data and independent controls are available.
2. The current `OpenFOAMTransportSnap` stores scalar summaries such as velocity, wall shear and residence time; it is not a spatial CFD field interchange. A mesh-aware field schema and conservative mapping are required for genuine spatial coupling.
3. The current `PHREEQCSpeciationSnap` stores a limited set of scalar concentrations/speciation values. It is not a PHREEQC input/output adapter and does not yet represent full species vectors, solution definitions, kinetic reactions, mineral inventories or charge/element balance audits.
4. The current core–multishell and emulsion tests verify internal/synthetic numerical relations with prescribed inputs. They do not validate the physical models against Ghosh's or Jack's experiments.
5. Solver integration claims must remain false until external solver repositories, versioned adapters, test cases, and empirical/benchmark validation have actually been completed.

## Recommended repository next steps

1. Add versioned coupling-schema documentation and JSON examples; add independent schema/unit/time/mesh/compartment validation tests.
2. Add a minimal offline OpenFOAM export/import fixture and a PHREEQC speciation fixture; record exact solver versions and input decks.
3. Add conservative scalar transport test cases with known analytical/reference solutions.
4. Add separate, optional Julia packages/modules for matrix mechanics, nutrient quota, QS/QQ topology, EV production and matrix-dependent diffusivity.
5. Add an experiment-data protocol and validation report template; do not populate with fabricated measurements.
6. Only after these pass, open focused PRs in the external OpenFOAM and PHREEQC solver repositories.

## Decision

**Proceed with modular interfaces and one-way snapshot coupling first.** Keep biological and matrix constitutive logic in Julia, spatial transport in OpenFOAM, aqueous speciation in PHREEQC, and FCS as an observation layer. Promote two-way multiphysics coupling only when data support the constitutive feedbacks and conservation/validation tests pass.
