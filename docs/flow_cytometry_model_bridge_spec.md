# Flow cytometry as the observation layer for coupled flow-geochemistry-biofilm models

**Status:** standalone observation-model interface specification and calibrated verification suite (`MODEL_ADAPTER_VERSION = "0.3.0"`, `ENGINE_VERSION = "0.3.0"`), verified with 177/177 local Julia unit assertions in `code/test_FlowCytometryObservationModel.jl` and 687/687 local Julia unit assertions in `code/test_MicrofluidicClassificationEngine.jl` (864/864 combined). Not yet coupled into external OpenFOAM or PHREEQC production solver repositories (`integrated_into_existing_openfoam_phreeqc_solvers: false`).

## 1. Scientific rationale and literature anchors

Flow cytometry adds high-throughput, single-cell population measurements to models whose primary forward state is fluid transport, aqueous chemistry, or bulk reaction rates. Four literature threads ground this specification:

1. **Explicit measurement and bead-calibration models:** Bedekar et al. (*Cytometry Part A*, 2025, https://doi.org/10.1002/cyto.a.24954) formulate a probabilistic per-event uncertainty model using calibration beads to separate biological or particle variability from instrument response, labeling variation, background signals, and measurement noise.
2. **Calibrating spatially structured microbial models:** Yip et al. (*PLOS Computational Biology* 18(10):e1010533, 2022, https://doi.org/10.1371/journal.pcbi.1010533) review single-cell observations, summary-statistic selection, calibration, and validation for microbial communities, cautioning that matching observed distributions does not uniquely identify the underlying biological mechanism (equifinality).
3. **Biofilm extraction, disaggregation, and matrix interference:** Kerstens et al. (2015, https://doi.org/10.1007/s12223-015-0400-4) and Grainha et al. (2020, https://doi.org/10.3390/antibiotics9110741) show that cytometric biofilm quantification depends critically on extraction recovery, disaggregation without selective lysis, and strain- or matrix-specific staining controls.
4. **Separating gating from absolute cell counting and coupled transport:** Preprints on *Measurement Quality Metrics to Improve Absolute Microbial Cell Counting* (2025) and *A Conservative Micro-Continuum-Cellular Automaton Method for Multispecies Biofilm Dynamics in Complex Flows* (2026) reinforce that total cells, viable cells, and events passing a fluorescence gate are distinct statistical quantities, and that mathematical coupling between flow and biofilm dynamics requires independent empirical validation.

**Core separation principle:** Scientific validity depends on keeping three layers distinct:
- the **latent biological state** being modeled in a specific compartment,
- the **measurement process** that maps latent cells to gated flow-cytometry events and bead ratios, and
- the **independent validation evidence** (held-out vessels/time points, endpoint microscopy, Raman, SEM/EDS, XRD, and coupon pit-depth measurements).

## 2. Responsibility by component

| Component | Forward-model responsibility | Outputs consumed by the coupling layer |
|---|---|---|
| OpenFOAM (`OpenFOAMTransportSnap`, `OpenFOAMMeshExchange`) | Velocity/pressure, wall shear, residence times, spatial transport fields, boundary patch fluxes, and detachment or deposition cues | `u(x,t)`, `wall_shear_pa`, `residence_time_s`, `internal_field`, `boundary_patches`, conservative volume-weighted compartment mapping |
| PHREEQC (`PHREEQCSpeciationSnap`, `PHREEQCSpeciesAdapter`) | Aqueous speciation, activity coefficients, charge/element balance, and mineral equilibria | `pH`, `pe`, `eh_volts`, `ionic_strength_M`, `species_molalities_mol_kgw`, `mineral_phases` ($\mathrm{SI} = \log_{10}(\mathrm{IAP}/K_{\mathrm{sp}})$) |
| Julia biological/reaction model (`BiofilmMultiphysics`) | Couple chemical/transport drivers to biofilm matrix mechanics (`PGA`, `EPS`, `Curli`, `Cellulose`), heterogeneous diffusivity $D_{\mathrm{eff}}(x,t;M,r,q)$, Droop quota $Q$, QS/EV signaling, and clonal subpopulation heterogeneity | `CellStatePrediction(total_cells_per_ml, (f_intact, f_compromised, f_dead))` per compartment and time point |
| Flow-cytometry adapter (`FlowCytometryObservationModel`) | Transform latent compartment states into expected event counts, gate probabilities, MESF/size calibrations, and likelihoods | `expected_cell_events`, `predicted_gate_probabilities`, `joint_loglikelihood`, `concentration_uncertainty` |
| Inference and validation layer (`EmpiricalValidationProtocol`) | Compare predictions with observed data, audit identifiability/equifinality, detect PSI domain shift, and enforce synthetic vs empirical generalization boundaries | `recover_latent_state_mle`, `check_parameter_identifiability`, `evaluate_equifinality`, `posterior_predictive_check`, `evaluate_empirical_generalization_readiness` |

**Important:** Flow cytometry is not a spatial sensor. Do not infer pore location or local EPS cross-link density from a cytometry sample alone. Pair it with endpoint microscopy/confocal imaging and planned Raman, SEM/EDS, and XRD observations. Every `FCMSampleSummary` records its `sample_origin` (`:effluent`, `:planktonic`, or `:detached_biofilm`), and `recover_latent_state_mle` rejects pooling samples across distinct compartments.

## 3. Priority 1: Observation-model specification and volume-convention consistency

### 3.1 Physical sampling protocol and volume conventions

Let $C$ (`total_cells_per_ml`) be the latent cell concentration in the sampled compartment, $r \in (0, 1]$ (`extraction_recovery`) be the independently validated fraction of cells recovered during sampling/detachment, and $\eta \in (0, 1]$ (`detection_efficiency`) be the independently validated fraction of recovered cells that enter the gated event population.

To prevent systematic misestimation between laboratory dilution protocols and the likelihood model, `FCMSampleSummary` explicitly records `volume_convention`:

1. `:aliquot_post_dilution` (default):
   - The extracted suspension is diluted by `dilution_factor` $d \ge 1$.
   - An assay volume $V = \text{sample\_volume\_ml}$ of the diluted suspension is placed into the cytometry tube (or $V$ denotes the initial sample volume before dividing by $d$), and $B_{\text{added}} = \text{beads\_added}$ counting beads are added to that diluted tube.
   - The effective undiluted original sample volume represented in the bead-spiked tube is:
     $$V_{\text{eff}} = \frac{V}{d}$$
2. `:undiluted_equivalent`:
   - $V = \text{sample\_volume\_ml}$ is already recorded as the net undiluted original sample volume present in the bead-spiked tube, so $V_{\text{eff}} = V$.
3. `:beads_co_diluted`:
   - $B_{\text{added}}$ counting beads are added directly to $V = \text{sample\_volume\_ml}$ of undiluted extract *before* co-diluting both cells and beads by $d$. Because cells and beads are diluted equally, the cell-to-bead ratio in the tube represents the full undiluted volume $V_{\text{eff}} = V$.

### 3.2 Exact consistency between concentration estimator and count likelihood

When the cytometer aspirates an unknown fraction of the assay tube and detects $B_{\text{events}}$ counting beads, the aspirated fraction of the bead-spiked tube is $B_{\text{events}} / B_{\text{added}}$. The expected number of detected gated cell events is (`expected_cell_events`):

$$\lambda(C) = C \, \eta \, V_{\text{eff}} \, r \, \frac{B_{\text{events}}}{B_{\text{added}}}$$

Under `:aliquot_post_dilution` ($V_{\text{eff}} = V / d$), this is:

$$\lambda(C) = \frac{C \, \eta \, V \, r \, B_{\text{events}}}{B_{\text{added}} \, d}$$

The Poisson count log-likelihood (`count_loglikelihood`) for $N_{\text{obs}} = \sum_{i=1}^3 n_i$ total gated cell events is:

$$\ell_{\text{count}}(C) = N_{\text{obs}} \log \lambda(C) - \lambda(C) - \log(N_{\text{obs}}!)$$

Differentiating $\ell_{\text{count}}(C)$ with respect to $C$ yields the score equation:

$$\frac{\partial \ell_{\text{count}}}{\partial C} = \frac{N_{\text{obs}}}{C} - \frac{\eta \, V_{\text{eff}} \, r \, B_{\text{events}}}{B_{\text{added}}} = 0 \quad \Longleftrightarrow \quad \hat{C}_{\text{MLE}} = \frac{N_{\text{obs}}}{B_{\text{events}}} \cdot \frac{B_{\text{added}}}{V_{\text{eff}} \, r \, \eta}$$

which is identical to `absolute_cells_per_ml(sample, calibration)`. The function `verify_concentration_likelihood_consistency(sample, calibration)` verifies both $\lambda(\hat{C}_{\text{MLE}}) = N_{\text{obs}}$ and $\left.\frac{\partial \ell_{\text{count}}}{\partial C}\right|_{C=\hat{C}_{\text{MLE}}} = 0$.

### 3.3 Conditional gate composition and overdispersed extensions

Conditional on total gated events $N_{\text{obs}}$, the 3-gate count vector $(n_1, n_2, n_3)$ is modeled by a Multinomial likelihood (`gate_loglikelihood`) with gate probabilities:

$$p = \frac{M_{\text{conf}} f}{\mathbf{1}^\top M_{\text{conf}} f}$$

where $M_{\text{conf}}$ is the $3 \times 3$ column-normalized gate confusion matrix and $f = (f_1, f_2, f_3)^\top$ is the latent state fraction vector. The joint log-likelihood (`joint_loglikelihood`) is $\ell_{\text{joint}} = \ell_{\text{count}} + \ell_{\text{gate}}$, avoiding duplicate counting of total events.

When biological or technical replicates exhibit variance exceeding Poisson-Multinomial shot noise, the module provides:
- `negative_binomial_count_loglikelihood(sample, state, calibration; dispersion_k)` with mean $\lambda$ and variance $\lambda + \lambda^2 / k$ (converging to `count_loglikelihood` as $k \to \infty$), and
- `dirichlet_multinomial_gate_loglikelihood(sample, state, calibration; concentration_alpha0)` with concentration $\alpha_0 > 0$ (converging to `gate_loglikelihood` as $\alpha_0 \to \infty$).

## 4. Priority 2: Synthetic-data simulation and parameter recovery

- `simulate_fcm_sample` generates synthetic bead counts and 3-gate cell counts under the exact forward observation model (`_rand_poisson` using exact additive Knuth decomposition for $\lambda \le 64$ and Atkinson's exact logistic rejection method for $\lambda > 64$, plus `_rand_multinomial3`).
- `recover_latent_state_mle` recovers the maximum-likelihood `CellStatePrediction` across replicate samples from a single compartment using exact closed-form Poisson exposure pooling for $\hat{C}_{\text{MLE}}$ and rank-deficient-safe simplex-projected Richardson-Lucy / EM iteration for $\hat{f}_{\text{MLE}}$ on the 3-simplex $\Delta^2$.

## 5. Priority 3: Calibration protocol (MESF, bead size, and gate controls)

1. **Fluorophore MESF calibration (`MESFCalibrationCurve`, `fit_mesf_calibration`, `predict_mesf`, `inverse_mesf`):**
   - Fits $\log_{10}(\text{MESF}) = a \log_{10}(\text{FL}) + b$ across $\ge 3$ calibration bead populations in channels such as `:FL1` ($\log_{10}(\text{MESF}) = 0.94 \log_{10}(\text{FL1}) + 1.63$, $R^2 > 0.999$) and `:FL2` ($\log_{10}(\text{MESF}) = 0.90 \log_{10}(\text{FL2}) + 1.71$, $R^2 = 0.997$).
2. **Bead size and scatter-area calibration (`BeadSizeCalibration`, `fit_bead_size_calibration`, `predict_equivalent_diameter_um`):**
   - Relates optical/scatter area ($\mu\text{m}^2$) to spherical cross-sectional area $\pi (D/2)^2 = a \cdot \text{Area} + b$ (for example $\pi (D/2)^2 = 1.26 \cdot \text{Area} - 5.67$, $R^2 > 0.999$).
3. **Independent gate-confusion and detection-efficiency estimation (`estimate_gate_calibration`):**
   - Estimates the $3 \times 3$ confusion matrix $M_{\text{conf}}$ from single-state positive controls (intact-like, compromised-like, dead-like) after subtracting `clay_blank_counts` from a clay-only matrix blank (requiring positive net biological events above the blank in every control column before applying pseudocount smoothing), and estimates `detection_efficiency` from spiked vs. recovered control cell counts (enforcing per-control recovery $\le 100\%$).

## 6. Priority 4: Uncertainty propagation and identifiability / equifinality analysis

- **Uncertainty propagation (`concentration_uncertainty`):** Propagates Poisson cell-count shot noise ($1/N_{\text{obs}}$), bead-count shot noise ($1/B_{\text{events}}$), and relative standard deviations in extraction recovery (`rel_sd_recovery`), detection efficiency (`rel_sd_efficiency`), and bead addition (`rel_sd_beads_added`) into a standard error and 95% confidence interval (`ConcentrationUncertainty`) that envelopes both the log-normal delta-method interval and the exact Garwood / Wilson-Hilferty Poisson count interval at low event counts ($N_{\text{obs}} \ge 1$). For zero-event samples ($N_{\text{obs}} = 0$), it reports the exact one-sided 97.5% Poisson upper limit ($-\log(0.025) \approx 3.6889$ events scaled by the volume/bead factor and multiplicative uncertainty).
- **Structural and numerical identifiability (`check_parameter_identifiability`):**
  - Audits the determinant and Frobenius condition number $\kappa_F(M_{\text{conf}})$ of the $3 \times 3$ confusion matrix.
  - Flags the structural scale collinearity of $(C, r, \eta)$: because count expectations depend only on the product $C \cdot r \cdot \eta$, latent concentration $C$ is unidentifiable unless $r$ and $\eta$ are fixed by independent controls.
- **Equifinality testing (`evaluate_equifinality`):** Compares candidate biological states or mechanisms (`state_a`, `state_b`) via $|\Delta \ell_{\text{joint}}|$, gate KL divergence, and exact joint Poisson-Multinomial KL divergence, requiring both finite log-likelihood closeness ($|\Delta \ell_{\text{joint}}| \le \text{loglik\_threshold}$) and distributional closeness ($\text{joint\_kl\_divergence} \le \text{kl\_threshold}$) before declaring two states equifinal.

## 7. Priority 5: Held-out validation and posterior-predictive checks

- `split_train_holdout` partitions vessels or time points into training and held-out validation subsets (rejecting out-of-bounds or duplicate holdout indices).
- `posterior_predictive_check` evaluates out-of-sample predictions on held-out observations, reporting `total_heldout_loglikelihood`, `mean_count_abs_z`, Pearson `gate_chi2_statistic` (returning `Inf` if positive counts are observed in a zero-expectation gate), symmetric 95% predictive count coverage (`empirical_coverage_95`), and the Freeman-Tukey discrepancy `posterior_predictive_p_value`.

## 8. Priority 6: Three stakeholder tracks and solver interface contracts

The software contract (`StakeholderTrackContract`, `OpenFOAMTransportSnap`, `PHREEQCSpeciationSnap`, `OpenFOAMMeshExchange`, `PHREEQCSpeciesAdapter`, `validate_solver_observation_alignment`) enforces explicit boundaries across the three stakeholder tracks:

1. **Track 1 (`:track1_nanobio` - Nano-bio and single-cell actuation, exploratory):**
   - Uses Julia reaction-diffusion/single-cell state modeling plus the flow-cytometry observation operator, with OpenFOAM or PHREEQC included only when the specific apparatus requires fluid mechanics or aqueous speciation.
   - Boundary: a transferable computational architecture does not validate nanocarrier uptake, actuation efficacy, or biological outcomes without independent imaging and controls.
2. **Track 2 (`:track2_clay_corrosion` - Clay-microbes-metal, 24-vessel pilot, validation required):**
   - Uses OpenFOAM for transport/shear, PHREEQC for aqueous speciation and mineral equilibria, Julia for biological state dynamics, and the cytometry adapter as an exploratory secondary observer on sampled populations.
   - Boundary: `StakeholderTrackContract` requires `primary_endpoint == "maximum_pit_depth_per_coupon"`. Cytometry characterizes sampled populations (`:effluent`, `:planktonic`, `:detached_biofilm`) and cannot displace coupon pit depth or establish pore-scale biofilm structure on its own.
3. **Track 3 (`:track3_bitumen_rheology` - Bitumen emulsion and interfacial rheology, separate validation track):**
   - Uses Julia for constitutive-law fitting and uncertainty analysis and OpenFOAM for multiphase/interfacial flow, validated against Dynamic Shear Rheometer (DSR) and moisture-susceptibility data.
   - Boundary: `StakeholderTrackContract` and `build_production_audit_record` reject attaching flow-cytometry observation models or PHREEQC speciation models to `:track3_bitumen_rheology`. Shared numerical infrastructure does not imply shared physical parameters.

## 9. Priority 7: Production acceptance checklist and epistemic status accounting

`EpistemicQuantity` and `build_production_audit_record` require every production audit bundle to record `MODEL_ADAPTER_VERSION`, the 64-character SHA-256 input digest, unique quantity names, and explicit separation of `:observed`, `:calibrated`, `:assumed`, and `:predicted` quantities:

- [x] Raw cytometry schema (`code/flow_cytometry_samples_template.csv`) validated against `REQUIRED_CSV_COLUMNS` via `validate_sample_csv_header`.
- [x] Calibration confusion matrix, detection efficiency, MESF curves, bead size curves, and extraction recovery estimated from independent control constructors (`estimate_gate_calibration`, `fit_mesf_calibration`, `fit_bead_size_calibration`).
- [x] Solver snapshots (`OpenFOAMTransportSnap`, `PHREEQCSpeciationSnap`, `OpenFOAMMeshExchange`, `PHREEQCSpeciesAdapter`) checked for exact vessel, time, and compartment alignment via `validate_solver_observation_alignment`.
- [x] Unit test suite (`code/test_FlowCytometryObservationModel.jl`) executed locally in Julia 1.13.1 (`177/177` passing assertions) covering concentration-likelihood consistency, synthetic-data recovery, zero/low counts, uncertainty propagation, equifinality, held-out posterior-predictive checks, OpenFOAM mesh field mapping, PHREEQC speciation balance, and empirical validation protocols.
- [x] All outputs explicitly tagged as `:observed`, `:calibrated`, `:assumed`, or `:predicted` via `EpistemicQuantity` and `ProductionAuditRecord`.

## 10. Microfluidic Cell Classification Workbench and Shared 3D Visualization Engine

`code/MicrofluidicClassificationEngine.jl` (`ENGINE_VERSION = "0.3.0"`) extends the standalone observation-model adapter into a complete Julia-first microfluidic single-cell classification engine, biofilm multiphysics coupling layer, and interactive browser workbench (`workbench/microfluidic_cytometry_workbench.html`), verified with `687/687` passing Julia unit assertions (`julia --startup-file=no code/test_MicrofluidicClassificationEngine.jl`) and cross-language numerical verification against an independent Python reference implementation (`code/verify_against_python_reference.py`).

### 10.1 Modular Julia architecture

| Submodule | Responsibility | Key types and functions |
|---|---|---|
| `FCSIO` | Standards-compliant ISAC FCS 3.1 binary file reading/writing, channel metadata (`$PnN`, `$PnS`, `$PnR`), SHA-256 provenance, and microfluidic sensor/trajectory CSV ingestion | `FCSChannelMetadata`, `FCSFileProvenance`, `FCSDataset`, `SensorTrajectoryRow`, `read_fcs_file`, `write_fcs_file`, `load_microfluidic_sensor_csv`, `generate_synthetic_nanobio_fcs_dataset` |
| `CytometryAnalysis` | Transformations (`linear`, `log10`, `arcsinh`, biexponential `logicle`), spectral spillover unmixing ($E_{\text{comp}} = E_{\text{raw}} S^{-1}$), quality control, hierarchical gates, and GatingML/JSON serialization | `LinearTransform`, `Log10Transform`, `ArcsinhTransform`, `BiexponentialLogicleTransform`, `SpilloverMatrix`, `apply_compensation`, `run_quality_control`, `RectangleGate`, `PolygonGate`, `EllipseGate`, `RangeGate`, `QuadGate`, `GatingHierarchy`, `evaluate_hierarchy` |
| `CellClassifier` | Single-cell feature extraction, calibrated supervised classification across the 4 Nano-Bio Lab target tasks, confidence abstention (`:defer_for_review`), strict group-aware splitting (`:chip_id`, `:run_id`, `:sample_id`, `:biological_replicate`), PSI domain-shift detection, and empirical generalization auditing | `MicrofluidicCellClassifier`, `SingleCellPrediction`, `ClassifierValidationReport`, `train_cell_classifier`, `predict_cells`, `group_aware_split`, `verify_no_group_leakage`, `group_cross_validate`, `detect_feature_domain_shift`, `audit_classifier_empirical_generalization` |
| `MicrofluidicBridge` | Joins cytometry events, hierarchical gates, classifier predictions, micropore transit trajectories ($\Delta t_{\text{transit}}$, blockage ratio $\Delta I / I_0$, deformability index), and `OpenFOAMMeshExchange`/`PHREEQCSpeciesAdapter` to `FlowCytometryObservationModel` | `MicrofluidicAssayCondition`, `MicroporeTransitEvent`, `JoinedMicrofluidicCellRecord`, `MicrofluidicBridgeReport`, `join_cytometry_and_microfluidics`, `summarize_to_fcm_sample`, `bridge_to_observation_model`, `export_reproducible_analysis_bundle` |
| `SharedVisualizationPhysics` | Domain-separated 3D wireframe-over-core computational physics for Track 1 (Dr. Santaneel Ghosh) and Track 3 (Iain Jack) | `CoreMultishellSpec`, `solve_core_multishell_rd`, `EmulsionDropletSpec`, `solve_emulsion_interfacial_droplet` |
| `BiofilmMultiphysics` | Switchable constitutive matrix mechanics, heterogeneous diffusivity, 1D reactive transport, Droop quota, QS/EV signaling, electrokinetic AHL degradation, polymicrobial crosstalk, and clonal subpopulation heterogeneity (Milestones M3 & M4) | `BiofilmHypothesisSwitches`, `BiofilmMatrixComposition`, `evaluate_biofilm_swelling_mechanics`, `compute_effective_diffusivity`, `solve_1d_reactive_transport_biofilm`, `evaluate_qs_signal_degradation`, `evaluate_polymicrobial_crosstalk`, `simulate_biofilm_regulatory_trajectory`, `subpopulations_to_fcm_state` |

### 10.2 Domain-separated 3D visualization physics

1. **Track 1 (Dr. Santaneel Ghosh - Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification):**
   - `solve_core_multishell_rd` solves the conservative multi-stratum radial spherical reaction-diffusion equation across concentric strata $\Omega_1 \dots \Omega_L$ ($0 < R_0 < R_1 < R_2 \dots < R_L$) with an octant cutaway and surrounding wireframe isosurface, coupling released payload $F_{\text{rel}}(t)$ and microfluidic channel exposure $C_{\text{exp}}(t)$ to single-cell classification fractions.
   - **Epistemic boundary:** `CoreMultishellSpec` enforces `track == :track1_nanobio` and `epistemic_status == :predicted`, labeling the shells as simulated concentration fields rather than observed physical layers unless independently verified by imaging.
2. **Track 3 (Iain Jack - Interface-Resolved Emulsion Dynamics: From Constitutive Rheology to Multiphase Flow):**
   - `solve_emulsion_interfacial_droplet` models bitumen/double-emulsion droplet deformation $D = (L - B)/(L + B)$ under shear rate $\dot{\gamma}$, capillary number $\text{Ca} = \mu_c \dot{\gamma} a_0 / \sigma(T)$, Boussinesq-Scriven temperature-dependent interfacial shear/dilatational viscosity $\mu_s(T), \kappa_s(T)$, internal twin-vortex circulation, and hydrodynamic pressure field $P/\text{Pa}$ across dimensionless times $t/\tau_\eta \in [1.0, 15.0]$.
   - **Boundary:** `EmulsionDropletSpec` enforces `track == :track3_bitumen_rheology` and `has_biological_release_shells == false`, reusing 3D wireframe/isosurface rendering infrastructure without importing biological release shell assumptions.

## 11. Multiphysics Solver Coupling and Switchable Biofilm Milestones (M1-M5)

Grounded in `docs/biofilm_openfoam_phreeqc_coupling_plan_2026-10-30.md` and the October 16-30, 2026 biofilm and quorum-sensing literature shortlist, version `0.3.0` implements five modular coupling milestones while keeping external OpenFOAM and PHREEQC production repositories decoupled until empirical benchmarks are completed:

- **M1 (Mesh-aware OpenFOAM transport exchange):** `OpenFOAMMeshExchange` and `OpenFOAMBoundaryPatch` parse and serialize OpenFOAM v2.0 ASCII `internalField` and `boundaryField` dictionaries, enforce volume-weighted conservative field mapping (`map_openfoam_field_conservative`) across mesh-to-compartment partitions of unity, and audit grid Courant/Peclet stability (`audit_openfoam_transport_stability`).
- **M2 (Full-species PHREEQC geochemical adapter):** `PHREEQCSpeciesAdapter` and `PHREEQCMineralPhase` parse and serialize PHREEQC selected-output TSV/CSV tables and audit aqueous charge neutrality, ionic strength, Nernst $E_h\text{-}pe$ consistency, element-total stoichiometry, and mineral saturation indices (`audit_phreeqc_speciation_balance`).
- **M3 (Switchable biofilm constitutive layer):** `BiofilmMatrixComposition` tracks $M(x,t) = \{\text{PGA}, \text{EPS}, \text{Curli}, \text{Cellulose}\}$, separating PGA-driven hydration/swelling expansion from curli/EPS elastic modulus and thin-film wrinkling buckling thresholds (*PNAS*, Oct 2026), computing curli/size/charge-dependent spatially heterogeneous diffusivity $D_{\mathrm{eff}}(x,t; M, r, q)$ (*npj Biofilms*, Oct 2026), and solving 1D conservative reactive transport with localized bioelectronic delivery $S_i(x,t)$.
- **M4 (Switchable regulatory, EV, and subpopulation dynamics):** `BiofilmRegulatoryState` and `simulate_biofilm_regulatory_trajectory` couple Droop intracellular nutrient quota $Q$, nutrient/stress-modulated `agr`/RNAIII/$\alpha$-PSM/EV release (*mBio*, Oct 2026), electric-field/ROS AHL signal removal vs intact-biofilm shielding (Oct 17, 2026 study), 3-mode polymicrobial QS crosstalk (*Critical Reviews in Microbiology*, Oct 2026), and *S. aureus* `fruA/fruK` cell-wall-thickened vancomycin-tolerant vs metabolically reduced vs oxidative-stress-repair subpopulations (*ISME J*, Oct 2026).
- **M5 (Empirical generalization vs synthetic verification gate):** `EmpiricalValidationProtocol`, `evaluate_empirical_generalization_readiness`, `detect_feature_domain_shift`, and `audit_classifier_empirical_generalization` prevent synthetic classification scores from being reported as empirical generalization and require quenched imaging controls before claiming intracellular nanocarrier internalization.


