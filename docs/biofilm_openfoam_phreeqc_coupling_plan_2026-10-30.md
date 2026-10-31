# Research-to-Implementation Plan: Biofilm, Quorum Sensing & Multiphysics Solver Coupling

**Date:** 2026-10-30  
**Status:** implemented in `FlowCytometryObservationModel.jl` v0.3.0 and `MicrofluidicClassificationEngine.jl` v0.3.0 (`BiofilmMultiphysics` Module 6); external OpenFOAM/PHREEQC solver repositories remain unmodified (`integrated_into_existing_openfoam_phreeqc_solvers: false`).

## Executive decision

Advance the existing Julia-first flow-cytometry-model-bridge through explicit, conservative coupling interfaces to OpenFOAM and PHREEQC. Do not embed biological signaling, cell-state transitions, flow-cytometry likelihoods, or nanocarrier classification directly into either solver. Keep Julia as the biological/regulatory/constitutive orchestration layer, OpenFOAM as the spatial transport/hydrodynamic solver, PHREEQC as the aqueous speciation/mineral-equilibrium engine, and flow cytometry as an observation operator.

Begin with offline, one-way snapshot coupling and conservation/unit audits. Introduce two-way feedback only after exchange variables, units, timing, compartment definitions, boundary conditions, and mass/charge balances are verified.

This plan uses the October 23-30 and related October 15-19 research summaries supplied for this project. The findings below are experimental motivations and candidate model extensions; they do not provide universal kinetic constants or validate a unified model.

## Current repository baseline and v0.3.0 implementation

The `main` branch now contains:
- `code/FlowCytometryObservationModel.jl` (v0.3.0), with explicit volume conventions, count/gate likelihoods, synthetic observation simulation and recovery, calibration, uncertainty, identifiability/equifinality, held-out checks, scalar `OpenFOAMTransportSnap` and `PHREEQCSpeciationSnap` contracts, plus:
  - **Milestone M1 (`OpenFOAMMeshExchange`, `OpenFOAMBoundaryPatch`)**: mesh-aware OpenFOAM v2.0 ASCII field dictionary serialization/parsing (`serialize_openfoam_ascii_field`, `parse_openfoam_ascii_field`), volume-weighted conservative field mapping (`map_openfoam_field_conservative`), Courant/Peclet/nonnegativity transport stability audits (`audit_openfoam_transport_stability`), and reduction to compartment snapshots (`summarize_openfoam_mesh_to_snap`).
  - **Milestone M2 (`PHREEQCSpeciesAdapter`, `PHREEQCMineralPhase`)**: full-species PHREEQC selected-output TSV/CSV serialization/parsing (`serialize_phreeqc_selected_output`, `parse_phreeqc_selected_output`), aqueous charge neutrality, ionic strength, Nernst $E_h\text{-}pe$ consistency, element-total stoichiometry, and mineral saturation index ($\mathrm{SI} = \log_{10}(\mathrm{IAP}/K_{\mathrm{sp}})$) audits (`audit_phreeqc_speciation_balance`, `summarize_phreeqc_adapter_to_snap`).
  - **Milestone M5 (`EmpiricalValidationProtocol`, `evaluate_empirical_generalization_readiness`)**: explicit epistemic separation between internal software verification on synthetic data (`software_verification_passed`) and empirical generalization on held-out experimental runs (`empirical_generalization_established`), including surface-association vs intracellular internalization warnings for `:nanocarrier_uptake`.
- `code/MicrofluidicClassificationEngine.jl` (v0.3.0), with FCS 3.1 binary ingestion, cytometry processing/gating, group-aware single-cell classification, Population Stability Index (PSI) domain-shift detection (`detect_feature_domain_shift`, `audit_classifier_empirical_generalization`), microfluidic trajectory joining, shared domain-separated visualization physics (`solve_core_multishell_rd` and `solve_emulsion_interfacial_droplet`), and **Module 6 (`BiofilmMultiphysics`)** implementing Milestones M3 and M4 with independently switchable hypotheses (`BiofilmHypothesisSwitches`).
- Browser workbench at `workbench/microfluidic_cytometry_workbench.html`.

**Test interpretation:** the verification suites establish internal mathematical and software implementation behavior for the tested cases. The classification suite uses simulated events and synthetic labels; its high scores do not establish empirical classification performance on Dr. Ghosh's experimental samples. Likewise the shell, emulsion, and biofilm multiphysics simulations use prescribed illustrative parameters and do not establish measured physical behavior until calibrated against experimental datasets.

## Solver responsibilities

| Component | Owns | Exports to coupling layer |
|---|---|---|
| OpenFOAM | Flow/pressure, advection, fluid shear, residence time, spatial transport, and any explicitly implemented fluid/transport mechanics | velocity, pressure, shear, transport fluxes, spatial concentration fields, boundary fluxes and mesh/time metadata (`OpenFOAMMeshExchange`) |
| PHREEQC | Aqueous speciation, activity corrections, mineral equilibria, selected aqueous/mineral kinetics | pH, pe/Eh assumptions, ionic strength, dissolved species, saturation indices and reaction/source/sink terms (`PHREEQCSpeciesAdapter`) |
| Julia | Biological state, cellular nutrient quota, QS/QQ network, stress and phenotype transitions, EV production, EPS/PGA/Curli/Cellulose states, constitutive matrix relations, model orchestration, inference and uncertainty | state-dependent rates, matrix property fields, biological source/sink terms, spatially and temporally defined reaction terms (`BiofilmMultiphysics`) |
| Flow cytometry / imaging observer | Measurement and data processing, not physical transport | observed gate counts, calibrated concentrations, classifier outputs, measurement uncertainty and provenance (`FlowCytometryObservationModel`) |

Flow cytometry is not an in-situ spatial sensor. It cannot independently identify pore-scale location, local EPS cross-link density or spatial matrix morphology. Pair it with confocal/endpoint microscopy, diffusion measurements, spectroscopy, microscopy or other independent measurements as appropriate.

PHREEQC should not become a generic biological signaling engine. It should provide chemically meaningful aqueous/speciation calculations. QS production, perception, transcriptional regulation and EV biogenesis remain biological model processes in Julia unless a narrowly defined chemical reaction is deliberately represented in PHREEQC.

## Research-driven model extensions (October 16-30, 2026)

### A. Biofilm matrix swelling and mechanics - PNAS, October 2026

The supplied paper summary describes complementary roles for PGA (water uptake/swelling) and EPS/curli (matrix cross-linking/structural integrity) in *B. subtilis* biofilms. This motivates an explicitly testable matrix model (`BiofilmMatrixComposition`, `ThinFilmSwellingMechanicsSpec`, `evaluate_biofilm_swelling_mechanics`) with distinct state variables:
- local PGA concentration (`pga_kg_m3`) driving osmotic swelling pressure and hydration;
- local EPS (`eps_kg_m3`), curli (`curli_kg_m3`), and cellulose (`cellulose_kg_m3`) governing composite elastic modulus $E_{\mathrm{film}}$;
- Euler-Foppl thin-film critical wrinkling stress $\sigma_{\mathrm{crit}}$, wrinkling index, and morphological phase (`:flat_unswollen`, `:swollen_smooth_gel`, `:wrinkled_buckled_colony`, `:rigid_compact_mat`).

Julia owns the candidate constitutive relationships and biological production/degradation terms. OpenFOAM receives a deformation/permeability/transport update only when those fields are physically part of the chosen coupled problem. Do not transplant *B. subtilis* parameters into other organisms or clay/microfluidic systems.

### B. Nutrient-dependent quorum sensing, stress response, EV biogenesis, and clonal heterogeneity - mBio & ISME J, October 2026

Represent separately in `BiofilmRegulatoryState` and `BiofilmRegulatoryParams`:
- extracellular nutrient availability $N$ and decoupled Droop intracellular nutrient quota $Q$;
- QS signal abundance $A$ and abiotic/biotic degradation (`ElectrokineticQSSpec`, Oct 17, 2026 study showing rapid AHL removal in cell-free supernatant vs incomplete disruption in intact biofilms);
- three-mode polymicrobial QS crosstalk (`PolymicrobialCrosstalkSpec`, *Critical Reviews in Microbiology*, Oct 2026) separating direct receptor competition, chemical-effector mediation, and environment-mediated alkaline lactonolysis / redox shift;
- species- and strain-specific transcriptional topology (`QSRegulatoryTopology`: `:staphylococcus_agr_rnaiii_psm`, `:pseudomonas_pqse_rhlr`, `:pseudomonas_lasi_lasr`);
- differential modulation of RNAIII vs $\alpha$-PSMs under nutrient/oxidative stress, with $\alpha$-PSM-driven extracellular vesicle ($EV$) biogenesis (*mBio*, Oct 2026);
- clonal *S. aureus* metabolic heterogeneity separating fructose-consuming `fruA/fruK` cell-wall-thickened vancomycin-tolerant cells, metabolically reduced cells, and oxidative-stress-repair cells (*ISME J*, Oct 2026).

### C. Matrix-dependent heterogeneous diffusion - UPEC study, npj Biofilms, October 2026

Move beyond a single uniform biofilm diffusivity when data support spatially varying transport (`HeterogeneousDiffusivitySpec`, `compute_effective_diffusivity`, `compute_spatial_diffusivity_field`):
- $D_{\mathrm{eff}}(x,t; M, r_{\mathrm{probe}}, q_{\mathrm{probe}})$ depends on matrix composition (specifically amyloid curli nanoporous sieving and anionic PGA/EPS charge binding), location, molecular size, charge, and ionic strength;
- distinguish diffusion from reversible adsorption/retention and chemical reaction;
- represent boundary conditions and source/sink terms explicitly in `solve_1d_reactive_transport_biofilm`.

### D. Localized delivery / bioelectronic perturbation - October 28, 2026 study

Represent spatially localized delivery as a source term $S_i(x,t)$ (`LocalizedDeliverySourceSpec`, `evaluate_localized_source_term`) with independently specified units, spatial support, pulse duration, and duty cycle.

### E. Species-specific quorum-sensing topology - PqsE promoter study, October 2026

Model regulatory network topology (`QSRegulatoryTopology`) separately from parameter values. A species-specific missing/added regulatory edge (such as `PqsE` chaperone coupling to `RhlR` in *P. aeruginosa*) cannot always be represented by merely changing a rate coefficient.

### F. Nutrient quota and bioprocess kinetics - Droop-model study, October 2026

Represent extracellular substrate $N$ and intracellular quota $Q$ as separate variables (`enable_droop_intracellular_quota` in `BiofilmHypothesisSwitches`). Calibrate uptake, quota dynamics, growth and production against batch data before using continuous-culture or reactor-scale predictions.

## Coupling sequence

### M0 - Freeze contracts before touching solvers

For every exchanged field, specify:
- physical meaning and unit;
- spatial support (cell/face/volume/compartment);
- coordinate system and mesh identifier;
- vessel/chip/sample/compartment identity;
- timestamp, timestep and interpolation rule;
- sign convention and whether a value is concentration, flux, rate, fraction or state;
- epistemic label (`observed`, `calibrated`, `assumed`, `predicted`);
- source hash, solver version, configuration and provenance.

### M1 - Offline one-way snapshot coupling (`OpenFOAMMeshExchange`)

Export OpenFOAM transport fields and PHREEQC speciation snapshots to a versioned, machine-readable schema. Import to Julia and evaluate the biological model without feeding values back into either solver.

Acceptance:
- units and sign conventions pass tests;
- vessel/chip/time/compartment alignment is enforced;
- input/output hashes and solver versions are recorded;
- no hidden interpolation or extrapolation;
- volume-weighted conservative mapping (`map_openfoam_field_conservative`) and CFL/Peclet stability checks (`audit_openfoam_transport_stability`) pass.

### M2 - Geochemical speciation & heterogeneous transport (`PHREEQCSpeciesAdapter`)

Verify full species vectors, charge neutrality, element stoichiometry, Nernst $E_h\text{-}pe$ consistency, and mineral saturation indices (`audit_phreeqc_speciation_balance`), alongside bounded spatial diffusivity and conservative 1D reactive transport (`solve_1d_reactive_transport_biofilm`).

### M3 & M4 - Switchable biofilm constitutive and regulatory layers (`BiofilmMultiphysics`)

All 8 constitutive and regulatory mechanisms are independently switchable via `BiofilmHypothesisSwitches` so simpler baseline models and complex multiphysics hypotheses can be compared systematically.

### M5 - Empirical predictive validation (`EmpiricalValidationProtocol`)

Use held-out biological replicates, chips, vessels or time points -- never random event-level splits. Enforce `evaluate_empirical_generalization_readiness` and `audit_classifier_empirical_generalization` so synthetic software verification is never conflated with empirical generalization.

## Track-specific boundaries

### Track 1 - Dr. Ghosh / nanobio and microfluidic classification

Use Julia core-multishell reaction-diffusion and the microfluidic classifier, with OpenFOAM only when the channel/device fluid mechanics materially affect transport. PHREEQC is optional and should be included only if relevant aqueous speciation or mineral chemistry exists in the actual assay.

The spherical shells are predicted concentration fields, not observed physical layers unless validated by imaging. The classifier must be trained and evaluated on expert-labelled, assay-specific experimental data with held-out chips/runs before claiming empirical generalization. Distinguish fluorescence association/uptake from demonstrated intracellular internalization (`surface_association_only_warning`).

### Track 2 - Clay-microbes-metal pilot

Use OpenFOAM for transport/shear, PHREEQC for aqueous chemistry/mineral equilibria, Julia for biological/matrix dynamics, and flow cytometry as an exploratory secondary observer on the specific sampled compartment. Preserve maximum pit depth per coupon as the primary corrosion endpoint. Require clay-only matrix blanks, counting beads, extraction-recovery validation, controls for staining/disaggregation and sample-origin tracking.

### Track 3 - Iain Jack / bitumen emulsion and interfacial rheology

Keep the existing emulsion droplet/interface model and constitutive fitting separate. Use Julia for temperature-dependent constitutive relations/uncertainty and OpenFOAM for multiphase/interfacial flow as needed. Validate with DSR and moisture-susceptibility evidence. Do not attach biological release shells, flow-cytometry observation models or clay-microbe PHREEQC chemistry to this track. Share rendering, provenance and data-contract infrastructure only.

## Implementation cautions found in current source

1. The synthetic-data generator in `MicrofluidicClassificationEngine.jl` assigns feature distributions and labels by construction. Strong cross-validation performance therefore demonstrates recoverability under the generator, not real-world assay classification. `EmpiricalValidationProtocol`, `evaluate_empirical_generalization_readiness`, and `audit_classifier_empirical_generalization` enforce this distinction programmatically until real de-identified, expert-labelled data and independent controls are available.
2. `OpenFOAMTransportSnap` stores scalar summaries such as velocity, wall shear and residence time, whereas `OpenFOAMMeshExchange` and `map_openfoam_field_conservative` provide the mesh-aware field schema and volume-weighted conservative mapping required for spatial coupling.
3. `PHREEQCSpeciationSnap` stores scalar concentrations/speciation values, whereas `PHREEQCSpeciesAdapter` and `audit_phreeqc_speciation_balance` represent full species vectors, activity coefficients, mineral inventories, Nernst $E_h\text{-}pe$ consistency, water autoionization ($K_w$), and charge/element/SI balance audits.
4. The core-multishell, emulsion, and `BiofilmMultiphysics` tests verify internal/synthetic numerical relations and conservation laws with prescribed inputs. They do not validate the physical models against Ghosh's, Jack's, or clay-pilot wet-lab experiments.
5. Solver integration claims (`integrated_into_existing_openfoam_phreeqc_solvers`) must remain `false` until external solver repositories, versioned adapters, test cases, and empirical/benchmark validation have actually been completed.

## Recommended repository next steps

1. Maintain versioned coupling-schema documentation and independent schema/unit/time/mesh/compartment validation tests (`OpenFOAMMeshExchange`, `PHREEQCSpeciesAdapter`).
2. Pair the offline OpenFOAM ASCII export/import parser and PHREEQC selected-output TSV/CSV adapter with recorded solver versions and input decks from the Track 2 pilot.
3. Verify conservative scalar and 1D reactive transport test cases (`solve_1d_reactive_transport_biofilm`, `map_openfoam_field_conservative`) against analytical and mass-balance invariants.
4. Keep matrix mechanics, nutrient quota, QS/QQ topology, EV production, and matrix-dependent diffusivity independently switchable via `BiofilmHypothesisSwitches` in `BiofilmMultiphysics`.
5. Enforce `EmpiricalValidationProtocol` and `audit_classifier_empirical_generalization` on all experimental validation reports without fabricated measurements.
6. Only after wet-lab calibration and offline snapshot benchmarks pass, open focused PRs in the external OpenFOAM and PHREEQC solver repositories.

## Decision

**Proceed with modular interfaces and one-way snapshot coupling first.** Keep biological and matrix constitutive logic in Julia, spatial transport in OpenFOAM, aqueous speciation in PHREEQC, and FCS as an observation layer. Promote two-way multiphysics coupling only when data support the constitutive feedbacks and conservation/validation tests pass.
