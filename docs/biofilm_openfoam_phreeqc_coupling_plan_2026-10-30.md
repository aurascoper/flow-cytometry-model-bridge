# Research-to-Implementation Plan: Biofilm, Quorum Sensing & Multiphysics Solver Coupling

**Date:** 2026-10-31  
**Status:** implemented in `FlowCytometryObservationModel.jl` v0.4.0 and `MicrofluidicClassificationEngine.jl` v0.4.0 (`BiofilmMultiphysics` Module 6, Milestones M1-M6); external OpenFOAM/PHREEQC solver repositories remain unmodified (`integrated_into_existing_openfoam_phreeqc_solvers: false`).

## Executive decision

Advance the existing Julia-first flow-cytometry-model-bridge through explicit, conservative coupling interfaces to OpenFOAM and PHREEQC. Do not embed biological signaling, cell-state transitions, flow-cytometry likelihoods, or nanocarrier classification directly into either solver. Keep Julia as the biological/regulatory/constitutive orchestration layer, OpenFOAM as the spatial transport/hydrodynamic solver, PHREEQC as the aqueous speciation/mineral-equilibrium engine, and flow cytometry as an observation operator.

Begin with offline, one-way snapshot coupling and conservation/unit audits. Introduce two-way feedback only after exchange variables, units, timing, compartment definitions, boundary conditions, and mass/charge balances are verified.

This plan uses the October 23-30 and related October 15-19 research summaries supplied for this project, together with recent reference comparators (Menke, Doster & Maes Oct 2026 arXiv:2610.19057; Kim & Yao Oct 2026 arXiv:2610.01565; Granato et al. 2025 *PRX Life* 3:043008). The findings below are experimental motivations and candidate model extensions; they do not provide universal kinetic constants or validate a unified model.

## Current repository baseline and v0.4.0 implementation

The `main` branch now contains:
- `code/FlowCytometryObservationModel.jl` (v0.4.0), with explicit volume conventions, count/gate likelihoods, synthetic observation simulation and recovery, calibration, uncertainty, identifiability/equifinality, held-out checks, scalar `OpenFOAMTransportSnap` and `PHREEQCSpeciationSnap` contracts, plus:
  - **Milestone M1 & M6.2 (`OpenFOAMMeshExchange`, `OpenFOAMBoundaryPatch`)**: mesh-aware OpenFOAM v2.0 ASCII field dictionary serialization/parsing (`serialize_openfoam_ascii_field`, `parse_openfoam_ascii_field`), anisotropic cell dimensions $(\Delta x, \Delta y, \Delta z)$ and directional velocity vectors $(u_x, u_y, u_z)$, geometry-verified bounding-box overlap weights (`build_geometric_overlap_matrix`), volume-weighted conservative field mapping (`map_openfoam_field_conservative`), spatial reconstruction error ($L_1, L_2, L_\infty$), center-of-mass centroid shift, and boundary-flux audits (`evaluate_mesh_mapping_reconstruction_error`), directional vs isotropic $\sqrt[3]{V_{\mathrm{cell}}}$ Courant/Peclet stability checks (`audit_openfoam_transport_stability`), and reduction to compartment snapshots (`summarize_openfoam_mesh_to_snap`).
  - **Milestone M2 & M6.2 (`PHREEQCSpeciesAdapter`, `PHREEQCMineralPhase`)**: full-species PHREEQC selected-output TSV/CSV serialization/parsing (`serialize_phreeqc_selected_output`, `parse_phreeqc_selected_output`), thermodynamic database version and reference electrode (`:SHE`, `:Ag_AgCl_3M_KCl`, `:SCE`) tracking (`verify_phreeqc_database_compatibility`), molality ($\mathrm{mol/kgw}$) to molarity ($\mathrm{mol/L}$) conversion via water mass per liter and solution density (`convert_molality_to_molarity`), aqueous charge neutrality, ionic strength, temperature-corrected Nernst $E_h\text{-}pe$ and $K_w(T)$ consistency, element-total stoichiometry, mineral saturation index ($\mathrm{SI} = \log_{10}(\mathrm{IAP}/K_{\mathrm{sp}})$) vs kinetically verified precipitation (`audit_phreeqc_speciation_balance`), and the Menke, Doster & Maes (Oct 2026, arXiv:2610.19057) conservative retardation benchmark (`run_menke_2026_retardation_benchmark` within $0.6\%$ tolerance).
  - **Milestone M5 & M6.5 (`EmpiricalValidationProtocol`, `evaluate_empirical_generalization_readiness`)**: explicit epistemic separation between internal software verification on synthetic data (`software_verification_passed`) and empirical generalization on held-out experimental runs (`empirical_generalization_established`), including surface-association vs intracellular internalization warnings for `:nanocarrier_uptake`, multivariate OOD fraction gates, minority-class recall gates, and biological replicate stratification gates.
  - **Milestone M6.1 (`DimensionalQuantity`, `ParameterProvenanceEntry`, `ParameterProvenanceRegistry`)**: OpenFOAM 7-tuple $[M, L, T, \Theta, N, I, J]$ to SI 7-tuple $[\mathrm{m}, \mathrm{kg}, \mathrm{s}, \mathrm{A}, \mathrm{K}, \mathrm{mol}, \mathrm{cd}]$ conversion (`openfoam_dims_to_si_exponents`, `si_exponents_to_openfoam_dims`), PDE dimensional homogeneity auditing (`audit_dimensional_consistency`), and parameter provenance/strain transferability auditing (`audit_parameter_provenance_registry`, Gate `G0`).
- `code/MicrofluidicClassificationEngine.jl` (v0.4.0), with FCS 3.1 binary ingestion, cytometry processing/gating, group-aware single-cell classification, univariate PSI/Z plus multivariate Mahalanobis distance and covariance Frobenius shift OOD detection (`detect_feature_domain_shift`, `audit_classifier_empirical_generalization`), microfluidic trajectory joining, shared domain-separated visualization physics (`solve_core_multishell_rd` and `solve_emulsion_interfacial_droplet`), and **Module 6 (`BiofilmMultiphysics`)** implementing Milestones M3, M4, and M6 (`build_default_biofilm_provenance_registry`, `calibrate_and_audit_effective_diffusivity`, non-uniform 1D FV `solve_1d_reactive_transport_biofilm`, `compare_spatial_vs_multiplicative_ahl_shielding`, `RegulatoryNetworkGraph`, `validate_regulatory_network_applicability`, `audit_droop_quota_conservation`, `run_biofilm_ablation_matrix`, and `evaluate_validation_ladder`).
- Browser workbench at `workbench/microfluidic_cytometry_workbench.html` (v0.4.0).

**Test interpretation:** the verification suites (`262/262` in `test_FlowCytometryObservationModel.jl` and `805/805` in `test_MicrofluidicClassificationEngine.jl`, `1,067` total assertions) establish internal mathematical and software implementation behavior for the tested cases. The classification suite uses simulated events and synthetic labels; its high scores do not establish empirical classification performance on Dr. Ghosh's experimental samples. Likewise the shell, emulsion, and biofilm multiphysics simulations use prescribed illustrative parameters and do not establish measured physical behavior until calibrated against experimental datasets.

## Solver responsibilities

| Component | Owns | Exports to coupling layer |
|---|---|---|
| OpenFOAM | Flow/pressure, advection, fluid shear, residence time, spatial transport, and any explicitly implemented fluid/transport mechanics | velocity vectors, cell dimensions, pressure, shear, transport fluxes, spatial concentration fields, boundary fluxes and mesh/time metadata (`OpenFOAMMeshExchange`) |
| PHREEQC | Aqueous speciation, activity corrections, mineral equilibria, selected aqueous/mineral kinetics | database version, reference electrode, water mass per liter, pH, pe/Eh assumptions, ionic strength, dissolved species, saturation indices and reaction/source/sink terms (`PHREEQCSpeciesAdapter`) |
| Julia | Biological state, cellular nutrient quota, QS/QQ network graph $\mathcal{G}=(V,E)$, stress and phenotype transitions, EV production, EPS/PGA/Curli/Cellulose states, constitutive matrix relations, model orchestration, inference and uncertainty | state-dependent rates, matrix property fields, biological source/sink terms, spatially and temporally defined reaction terms (`BiofilmMultiphysics`) |
| Flow cytometry / imaging observer | Measurement and data processing, not physical transport | observed gate counts, calibrated concentrations, classifier outputs, measurement uncertainty and provenance (`FlowCytometryObservationModel`) |

Flow cytometry is not an in-situ spatial sensor. It cannot independently identify pore-scale location, local EPS cross-link density or spatial matrix morphology. Pair it with confocal/endpoint microscopy, diffusion measurements, spectroscopy, microscopy or other independent measurements as appropriate.

PHREEQC should not become a generic biological signaling engine. It should provide chemically meaningful aqueous/speciation calculations. QS production, perception, transcriptional regulation and EV biogenesis remain biological model processes in Julia unless a narrowly defined chemical reaction is deliberately represented in PHREEQC.

## Research-driven model extensions (October 16-31, 2026)

### A. Biofilm matrix swelling, constitutive maturity levels, and mechanics - PNAS Oct 2026 & Granato et al. PRX Life 2025

The supplied paper summary describes complementary roles for PGA (water uptake/swelling) and EPS/curli (matrix cross-linking/structural integrity) in *B. subtilis* biofilms. This motivates an explicitly testable matrix model (`BiofilmMatrixComposition`, `ThinFilmSwellingMechanicsSpec`, `evaluate_biofilm_swelling_mechanics`) tagged with constitutive maturity levels (`:level1_phenomenological_hypothesis`, `:level2_calibrated_constitutive`, `:level3_coupled_poroelastic_viscoelastic`):
- local PGA concentration (`pga_kg_m3`) driving osmotic swelling pressure and hydration;
- local EPS (`eps_kg_m3`), curli (`curli_kg_m3`), and cellulose (`cellulose_kg_m3`) governing composite elastic modulus $E_{\mathrm{film}}$;
- Euler-Foppl thin-film critical wrinkling stress $\sigma_{\mathrm{crit}}$, wrinkling index, poroelastic consolidation time, viscoelastic relaxed modulus, and morphological phase (`:flat_unswollen`, `:swollen_smooth_gel`, `:wrinkled_buckled_colony`, `:rigid_compact_mat`).

Julia owns the candidate constitutive relationships and biological production/degradation terms. OpenFOAM receives a deformation/permeability/transport update only when those fields are physically part of the chosen coupled problem. Do not transplant *B. subtilis* parameters into other organisms or clay/microfluidic systems (`ParameterProvenanceRegistry` enforces organism/strain tags).

### B. Nutrient-dependent quorum sensing, stress response, EV biogenesis, and clonal heterogeneity - mBio & ISME J, October 2026

Represent separately in `BiofilmRegulatoryState` and `BiofilmRegulatoryParams`:
- extracellular nutrient availability $N$ and decoupled Droop intracellular nutrient quota $Q$, audited for strict closed-system limiting-element conservation $N(t) + Q(t) X(t) = \mathrm{const}$ (`audit_droop_quota_conservation`);
- QS signal abundance $A$ and abiotic/biotic degradation (`ElectrokineticQSSpec`, Oct 17, 2026 study), separating direct electrochemical oxidation, ROS-mediated abiotic degradation, and biological enzymatic quorum quenching (`evaluate_qs_signal_degradation`), plus 1D spatial reaction-diffusion vs 0D multiplicative shielding comparison (`compare_spatial_vs_multiplicative_ahl_shielding`);
- three-mode polymicrobial QS crosstalk (`PolymicrobialCrosstalkSpec`, *Critical Reviews in Microbiology*, Oct 2026) separating direct receptor competition, chemical-effector mediation, and environment-mediated alkaline lactonolysis / redox shift;
- auditable regulatory network graph $\mathcal{G}=(V,E)$ (`RegulatoryNetworkGraph`, `build_species_regulatory_network`, `validate_regulatory_network_applicability`) enforcing strict species separation between *S. aureus* (`:staphylococcus_agr_rnaiii_psm`) and *P. aeruginosa* (`:pseudomonas_pqse_rhlr`, `:pseudomonas_lasi_lasr`) unless an explicit `:polymicrobial_co_culture` graph is constructed;
- differential modulation of RNAIII vs $\alpha$-PSMs under nutrient/oxidative stress, with $\alpha$-PSM-driven extracellular vesicle ($EV$) biogenesis (*mBio*, Oct 2026);
- clonal *S. aureus* metabolic heterogeneity separating fructose-consuming `fruA/fruK` cell-wall-thickened vancomycin-tolerant cells, metabolically reduced cells, and oxidative-stress-repair cells (*ISME J*, Oct 2026).

### C. Matrix-dependent heterogeneous diffusion & non-uniform 1D reactive transport - npj Biofilms Oct 2026 & Kim & Yao arXiv:2610.01565

Move beyond a single uniform biofilm diffusivity when data support spatially varying transport (`HeterogeneousDiffusivitySpec`, `compute_effective_diffusivity`, `calibrate_and_audit_effective_diffusivity`):
- $D_{\mathrm{eff}}(x,t; M, r_{\mathrm{probe}}, q_{\mathrm{probe}})$ depends on matrix composition (specifically amyloid curli nanoporous sieving and anionic PGA/EPS charge binding), location, molecular size, charge, and ionic strength;
- multi-probe calibration audits size-exclusion monotonicity, curli sieving monotonicity, cationic Donnan retardation, and Debye ionic-strength screening;
- `solve_1d_reactive_transport_biofilm` supports both uniform and non-uniform finite-volume cell widths $\Delta x_i$ with distance-weighted harmonic interface conductance $G_{i+1/2} = \frac{2 D_i D_{i+1}}{\Delta x_i D_{i+1} + \Delta x_{i+1} D_i}$.

### D. Localized delivery / bioelectronic perturbation - October 28, 2026 study

Represent spatially localized delivery as a source term $S_i(x,t)$ (`LocalizedDeliverySourceSpec`, `evaluate_localized_source_term`) with independently specified units, spatial support, pulse duration, and duty cycle.

### E. Controlled ablation matrix ($R_0 \dots R_5$ + pairwise interactions) & 6-gate validation ladder (`G0`-`G5`)

- `run_biofilm_ablation_matrix` evaluates 17 configurations ($R_0$, single-module $R_1 \dots R_5$, all 10 pairwise combinations $R_{ij}$, and $R_{\mathrm{full}}$), computing main effects $\Delta_i$, pairwise epistatic/synergistic interactions $\Delta_{ij} = Y_{ij} - Y_i - Y_j + Y_0$, and first-order variance sensitivity indices.
- `evaluate_validation_ladder` evaluates the 6-gate progression ladder (`G0_dimensional_provenance`, `G1_mesh_conservation_cfl`, `G2_phreeqc_thermodynamic_benchmark`, `G3_constitutive_transport_verification`, `G4_regulatory_graph_quota_audit`, `G5_empirical_generalization_gate`), ensuring `G5` remains `false` (`claim_boundary = :synthetic_numerical_verification_only`) on synthetic benchmarks.

## Coupling sequence

### M0 & M6.1 (Gate G0) - Freeze dimensional and parameter provenance contracts

For every exchanged field and model parameter, specify:
- physical meaning, unit string, and SI base exponent vector $[\mathrm{m}, \mathrm{kg}, \mathrm{s}, \mathrm{A}, \mathrm{K}, \mathrm{mol}, \mathrm{cd}]$ (`DimensionalQuantity`, `ParameterProvenanceEntry`);
- organism/strain applicability (`VALID_ORGANISM_STRAINS`) and calibration status (`audit_parameter_provenance_registry`).

### M1 & M6.2 (Gate G1) - Geometry-aware OpenFOAM snapshot coupling (`OpenFOAMMeshExchange`)

Export OpenFOAM transport fields and import into Julia with anisotropic directional CFL/Peclet stability checks (`audit_openfoam_transport_stability`), geometric bounding-box overlap weights (`build_geometric_overlap_matrix`), and spatial reconstruction/centroid-shift audits (`evaluate_mesh_mapping_reconstruction_error`).

### M2 & M6.2 (Gate G2) - Geochemical speciation & Menke 2026 benchmark (`PHREEQCSpeciesAdapter`)

Verify full species vectors, database compatibility (`verify_phreeqc_database_compatibility`), reference electrode convention, molality-to-molarity conversion (`convert_molality_to_molarity`), charge neutrality, element stoichiometry, Nernst $E_h\text{-}pe$ consistency, mineral saturation indices vs kinetic precipitation (`audit_phreeqc_speciation_balance`), and the Menke et al. (Oct 2026) retardation benchmark (`run_menke_2026_retardation_benchmark`).

### M3, M4, M6.3 & M6.4 (Gates G3 & G4) - Switchable biofilm constitutive and regulatory layers (`BiofilmMultiphysics`)

Verify multi-probe diffusivity invariants (`calibrate_and_audit_effective_diffusivity`), non-uniform 1D FV reactive transport (`solve_1d_reactive_transport_biofilm`), species-separated regulatory graph $\mathcal{G}=(V,E)$ (`validate_regulatory_network_applicability`), and closed-system Droop quota conservation (`audit_droop_quota_conservation`).

### M5 & M6.5 (Gate G5) - Empirical predictive validation (`EmpiricalValidationProtocol` & `ValidationLadderReport`)

Use held-out biological replicates, chips, vessels or time points -- never random event-level splits. Enforce `detect_feature_domain_shift` (including multivariate Mahalanobis OOD and covariance Frobenius shift), `evaluate_empirical_generalization_readiness`, `audit_classifier_empirical_generalization`, and `evaluate_validation_ladder`.

## Track-specific boundaries

### Track 1 - Dr. Ghosh / nanobio and microfluidic classification

Use Julia core-multishell reaction-diffusion and the microfluidic classifier, with OpenFOAM only when the channel/device fluid mechanics materially affect transport. PHREEQC is optional and should be included only if relevant aqueous speciation or mineral chemistry exists in the actual assay.

The spherical shells are predicted concentration fields, not observed physical layers unless validated by imaging. The classifier must be trained and evaluated on expert-labelled, assay-specific experimental data with held-out chips/runs before claiming empirical generalization. Distinguish fluorescence association/uptake from demonstrated intracellular internalization (`surface_association_only_warning`).

### Track 2 - Clay-microbes-metal pilot

Use OpenFOAM for transport/shear, PHREEQC for aqueous chemistry/mineral equilibria, Julia for biological/matrix dynamics, and flow cytometry as an exploratory secondary observer on the specific sampled compartment. Preserve maximum pit depth per coupon as the primary corrosion endpoint. Require clay-only matrix blanks, counting beads, extraction-recovery validation, controls for staining/disaggregation and sample-origin tracking.

### Track 3 - Iain Jack / bitumen emulsion and interfacial rheology

Keep the existing emulsion droplet/interface model and constitutive fitting separate. Use Julia for temperature-dependent constitutive relations/uncertainty and OpenFOAM for multiphase/interfacial flow as needed. Validate with DSR and moisture-susceptibility evidence. Do not attach biological release shells, flow-cytometry observation models or clay-microbe PHREEQC chemistry to this track. Share rendering, provenance and data-contract infrastructure only.

## Implementation cautions found in current source

1. The synthetic-data generator in `MicrofluidicClassificationEngine.jl` assigns feature distributions and labels by construction. Strong cross-validation performance therefore demonstrates recoverability under the generator, not real-world assay classification. `EmpiricalValidationProtocol`, `evaluate_empirical_generalization_readiness`, `audit_classifier_empirical_generalization`, and Gate `G5` of `evaluate_validation_ladder` enforce this distinction programmatically until real de-identified, expert-labelled data and independent controls are available.
2. `OpenFOAMTransportSnap` stores scalar summaries such as velocity, wall shear and residence time, whereas `OpenFOAMMeshExchange`, `build_geometric_overlap_matrix`, and `evaluate_mesh_mapping_reconstruction_error` provide the anisotropic mesh-aware field schema, conservative overlap mapping, and spatial reconstruction error audits required for spatial coupling.
3. `PHREEQCSpeciationSnap` stores scalar concentrations/speciation values, whereas `PHREEQCSpeciesAdapter`, `verify_phreeqc_database_compatibility`, `convert_molality_to_molarity`, and `audit_phreeqc_speciation_balance` represent full species vectors, database versions, reference electrodes, water mass per liter, mineral kinetic rates, and thermodynamic balance audits.
4. The core-multishell, emulsion, and `BiofilmMultiphysics` tests verify internal/synthetic numerical relations, literature-grounded reference comparators (Menke et al. Oct 2026), and conservation laws with prescribed inputs. They do not validate the physical models against Ghosh's, Jack's, or clay-pilot wet-lab experiments.
5. Solver integration claims (`integrated_into_existing_openfoam_phreeqc_solvers`) must remain `false` until external solver repositories, versioned adapters, test cases, and empirical/benchmark validation have actually been completed.

## Decision

**Proceed with modular interfaces, physics/biology calibration audits (M6 Gates G0-G5), and one-way snapshot coupling first.** Keep biological and matrix constitutive logic in Julia, spatial transport in OpenFOAM, aqueous speciation in PHREEQC, and FCS as an observation layer. Promote two-way multiphysics coupling only when data support the constitutive feedbacks and conservation/validation tests pass.

