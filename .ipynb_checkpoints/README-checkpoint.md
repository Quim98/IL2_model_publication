# Combining signaling models with sequencing data enables the prediction of variant/cell type-specific IL-2 responses

This repository contains the computational code used in the manuscript:

"Combining signaling models with sequencing data enables the prediction of variant/cell type-specific IL-2 responses"

It combines gene-specific RNA-to-protein abundance estimation, mechanistic models of IL-2 signaling, approximate Bayesian computation–sequential Monte Carlo (ABC-SMC) parameter inference, dose-response simulations, and in vivo analyses of immune cells and melanoma tumor microenvironments.

**Preprint:** 
**Code:** https://github.com/Quim98/IL2_model_publication
**Processed data and simulation outputs:**

---

## Reproducibility overview

This repository contains the modeling, inference, simulation, analysis, and plotting code. The scripts and notebooks use input files under `data/`, previously fitted parameters and computational outputs under `results/`, and save plots under `figures/`. Large processed datasets, model initial conditions, and simulation/inference outputs are hosted separately on Zenodo. o reproduce the figures, clone this repository, extract the Zenodo archive into the repository root while preserving its directory structure, create the supplied environment, and run the corresponding plotting notebook.

---

## Repository structure

The principal code is organized as follows:

```text
IL2_model_publication/
├── packages.yml                   # Conda environment specification
├── src/
│   ├── simulate_func.py           # Shared simulation, inference, and analysis functions
│   └── models/                   # PySB IL-2 model implementations
│       └── BEMPEG_trials/         # Model and utilities for BEMPEG simulations
│
├── pyabc_inference_WT.py          # ABC-SMC inference for WT IL-2
├── pyabc_inference_var.py         # ABC-SMC inference for engineered IL-2 variants
├── pyabc_BEMPEG_inference.py      # ABC-SMC inference for BEMPEG variants
├── pyabc_mutant_search.py        # Optimization of receptor binding for cell selectivity
├── exec_slopiness.py             # Sensitivity/sloppiness analysis of model fitting
├── exec_slopiness_target.py      # Sensitivity analysis for cell-selective targeting
├── exec_sim_invitro_fit.py       # In-vitro dose-response simulations
├── exec_sim_prior_post.py        # Simulations sampled from prior/posterior distributions
├── exec_CD8_cell_cycle.py        # Autocrine IL-2 signaling in CD8 T cells
├── exec_cytk_dict.py             # Immune Dictionary simulations
├── BEMPEG_RB_thres.py            # IL-2Rβ-abundance threshold analysis
├── BEMPEG_sim_invitro.py         # BEMPEG in-vitro simulations
├── BEMPEG_sim_tumor.py           # BEMPEG simulations in melanoma TME
├── BEMPEG_sim_tumor_optimized.py # Optimized IL-2 variant simulations in melanoma TME
├── scGen_model_training.py       # Batch correction of melanoma scRNA-seq data
│
├── Plots_linear_models.ipynb
├── Plots_linear_models_CPM.ipynb
├── Plot_slopiness_analysis.ipynb
├── Plot_ABC_SMC.ipynb
├── Plots_dose_response.ipynb
├── Mech_models_performance.ipynb
├── Null_model_IL2.ipynb
├── CD8_cell_cycle.ipynb
├── Plot_cytokine_dictionary.ipynb
├── IL2_mutants.ipynb
├── BEMPEG_datasets.ipynb
└── Plots_BEMPEG.ipynb
```

The directories below are supplied in the Zenodo archive and should be placed in the repository root:

```text
data/      # Processed expression, binding, signaling, and perturbation datasets
results/   # Fitted parameters, inference histories, simulations, sensitivity analyses
```

### Zenodo data structure

| Directory | Main contents used by the code |
| --- | --- |
| `data/expression/` | RNA/protein abundance datasets for fitting RNA-to-protein models and initializing IL-2 signaling simulations. |
| `data/expression/cell_cycle_CD8/` | Expression data for CD8⁺ T-cell cell-cycle/autocrine-signaling analyses. |
| `data/expression/melanoma/` | Processed melanoma single-cell expression data used for tumor-microenvironment simulations. |
| `data/melanoma/` | Melanoma expression data and scGen model files used by `scGen_model_training.py` (a distinct path in that script). |
| `data/lin_models/` | Trained RNA-to-protein model parameters, including the version fitted to single-cell CPM-normalized data. |
| `data/binding/` | IL-2 and variant receptor-binding parameters. |
| `data/signaling/` | Experimental IL-2 dose-response measurements, fitted EC50/amplitude data, mutant measurements, and Immune Dictionary inputs. |
| `data/ABC_SMC/BEMPEG_trials/` | BEMPEG experimental inputs and parameter bounds used for ABC-SMC fitting. |
| `results/fit_ABC_SMC/` | Fitted parameter tables and ABC-SMC outputs/databases used in downstream simulations and plots. |
| `results/immune_dict/` | Simulated pSTAT5 responses for Immune Dictionary cell types. |
| `results/BEMPEG_trials/` | BEMPEG and optimized IL-2 variant simulation outputs, including melanoma TME predictions. |
| `results/` | Other dose-response, CD8-autocrine, prior/posterior, and Hessian outputs. |

---

## Software environment

The complete computational environment is specified in `environment.yaml`.

We recommend using **mamba/Miniforge** to create the environment.

### 1. Clone the repository

```bash
git clone https://github.com/Quim98/IL2_model_publication.git
cd IL2_model_publication
```

### 2. Download the Zenodo dataset

Download the archive associated with:

**x**

Extract its contents directly into the repository root. After extraction, the repository should contain at least:


### 3. Create and activate the environment

```bash
mamba env create -f environment.yaml
mamba activate IL2_model
```

### 4. Run the notebooks

```bash
jupyter lab
```

---

## Main computational workflow

### 1. Estimating receptor and STAT5 protein abundance

The analysis uses five gene-specific RNA-to-protein models to estimate cellular abundances of **IL-2Rα (IL2RA), IL-2Rβ (IL2RB), common γ chain (IL2RG), STAT5A, and STAT5B** from RNA expression when protein copy numbers are unavailable. These abundances are used to define cell-type-specific initial conditions for signaling simulations.

- [`Plots_linear_models.ipynb`](Plots_linear_models.ipynb) — trains the five RNA-to-protein regression models, estimates protein abundance in the expression dataset, and generates model-fitting and validation plots.
- [`Plots_linear_models_CPM.ipynb`](Plots_linear_models_CPM.ipynb) — recalibrates gene-specific abundance models for single-cell RNA-seq expression normalized as counts per million (CPM), for use with the single-cell perturbation and melanoma datasets.

**Typical inputs:** `data/expression/whole_dataset.csv`, `data/expression/tissues_dataset.csv`, and `data/expression/immune_dict_dataset.tsv.gz`.

**Typical outputs:** `data/lin_models/linear_models.csv` and `data/lin_models/linear_models_IL2_CPM.csv`.

### 2. Mechanistic modeling of IL-2-induced STAT5 activation

The signaling models in `src/models/` use PySB to represent IL-2 binding to IL-2Rα, IL-2Rβ, and IL-2Rγ, formation of receptor complexes, and the resulting STAT5 activation. Shared code in [`src/simulate_func.py`](src/simulate_func.py) handles initialization, dose-response simulation, parameter inference, and supporting calculations.

The source includes `IL2s_parallel.py` and `IL2s_sequential.py`, used by different simulation and inference scripts. Simulations predict pSTAT5 responses as a function of cytokine dose, receptor-binding kinetics, and cellular protein abundances.

### 3. Parameter sensitivity and ABC-SMC inference

A sloppiness analysis characterizes which model parameters or parameter combinations most influence predictions. ABC-SMC is then used to infer kinetic parameters from measured IL-2 dose-response behavior, including engineered IL-2 variants.

- [`exec_slopiness.py`](exec_slopiness.py) — calculates a Hessian-based sloppiness/sensitivity analysis for model fitting.
- [`Plot_slopiness_analysis.ipynb`](Plot_slopiness_analysis.ipynb) — visualizes model-parameter sensitivity and receptor-binding sensitivity for cell-selective optimization.
- [`pyabc_inference_WT.py`](pyabc_inference_WT.py) — performs ABC-SMC parameter inference for wild-type IL-2.
- [`pyabc_inference_var.py`](pyabc_inference_var.py) — performs ABC-SMC parameter inference for non-WT IL-2 variants.
- [`exec_sim_prior_post.py`](exec_sim_prior_post.py) — generates dose-response predictions from parameters sampled from ABC-SMC prior and posterior distributions.
- [`Plot_ABC_SMC.ipynb`](Plot_ABC_SMC.ipynb) — visualizes parameter priors/posteriors, their predicted EC50 and response-amplitude distributions, and the receptor-binding optimization results.

**Typical inputs:** `data/binding/`, `data/signaling/`, `data/lin_models/`, `data/expression/`, and parameter-fitting tables under `results/fit_ABC_SMC/`.

**Typical outputs:** fitted parameters and inference databases under `results/fit_ABC_SMC/`, prior/posterior simulation files under `results/`, and Hessian-derived sensitivity outputs.

### 4. In-vitro IL-2 dose-response prediction and model evaluation

The fitted model is used to simulate pSTAT5 responses across IL-2 concentrations, cell types, and IL-2 variants and to compare predicted dose-response curves, EC50 values, and response amplitudes with experimental measurements.

- [`exec_sim_invitro_fit.py`](exec_sim_invitro_fit.py) — simulates fitted IL-2 variant dose-response curves in immune and cancer cell types.
- [`Plots_dose_response.ipynb`](Plots_dose_response.ipynb) — compares experimental and model-simulated dose-response curves.
- [`Mech_models_performance.ipynb`](Mech_models_performance.ipynb) — evaluates predictive accuracy across datasets, variants, and cell types, including training and test conditions.
- [`Null_model_IL2.ipynb`](Null_model_IL2.ipynb) — compares the mechanistic IL-2 model with regression-based alternatives trained using the same signaling-model inputs (IL-2 dose, receptor abundance, and STAT5 abundance).

**Typical outputs:** `results/simulations_IL2s_fit.csv`, performance summaries, and dose-response plots.

### 5. Independent biological validation

The model is tested against measurements that were not simply the original in-vitro dose-response fitting curves.

**Autocrine IL-2 signaling during the CD8⁺ T-cell cell cycle**

- [`exec_CD8_cell_cycle.py`](exec_CD8_cell_cycle.py) — predicts autocrine WT IL-2-induced pSTAT5 under different CD8⁺ T-cell cell-cycle conditions.
- [`CD8_cell_cycle.ipynb`](CD8_cell_cycle.ipynb) — compares these predictions with experimental pSTAT5 measurements and generates the corresponding plots.

**Immune Dictionary**

- [`exec_cytk_dict.py`](exec_cytk_dict.py) — simulates IL-2-induced pSTAT5 across immune cell types represented in the Immune Dictionary.
- [`Plot_cytokine_dictionary.ipynb`](Plot_cytokine_dictionary.ipynb) — relates model-simulated pSTAT5 abundance to the magnitude of IL-2-induced transcriptomic changes across cell types.

**Typical outputs:** `results/simulations_CD8cc.csv` and `results/immune_dict/simulations_cytk_dict.csv`.

### 6. BEMPEG and tumor-microenvironment analyses

BEMPEG (bempegaldesleukin) is analyzed through its active **1-PEG-IL-2 and 2-PEG-IL-2** species. The simulations compare their pSTAT5 activity across immune cells and the metastatic melanoma tumor microenvironment (TME), alongside predictions for a model-optimized IL-2 variant.

- [`pyabc_BEMPEG_inference.py`](pyabc_BEMPEG_inference.py) — fits receptor-binding parameters of BEMPEG variants using ABC-SMC.
- [`BEMPEG_sim_invitro.py`](BEMPEG_sim_invitro.py) — predicts in-vitro pSTAT5 responses for BEMPEG variants in CTLL-2 cells and selected immune cell populations.
- [`BEMPEG_RB_thres.py`](BEMPEG_RB_thres.py) — varies IL-2Rβ abundance to examine its influence on simulated pSTAT5 activation.
- [`scGen_model_training.py`](scGen_model_training.py) — trains an scGen model for batch correction of melanoma single-cell RNA-seq datasets.
- [`BEMPEG_datasets.ipynb`](BEMPEG_datasets.ipynb) — integrates melanoma datasets, estimates tumor concentrations of active BEMPEG species, examines CD8⁺ T-cell and Treg subpopulations, and produces UMAP and related dataset visualizations.
- [`BEMPEG_sim_tumor.py`](BEMPEG_sim_tumor.py) — predicts responses to 1-PEG-IL-2 and 2-PEG-IL-2 across cell populations in metastatic melanoma.
- [`BEMPEG_sim_tumor_optimized.py`](BEMPEG_sim_tumor_optimized.py) — simulates the model-optimized IL-2 variant across the same tumor-microenvironment context.
- [`Plots_BEMPEG.ipynb`](Plots_BEMPEG.ipynb) — generates the BEMPEG dose-response and tumor-microenvironment analysis plots, including comparisons with the optimized variant.

**Typical inputs:** `data/ABC_SMC/BEMPEG_trials/`, `data/expression/melanoma/`, `data/lin_models/`, and BEMPEG-specific fitted parameter tables.

**Typical outputs:** `results/BEMPEG_trials/` and corresponding plots under `figures/`.

### 7. Engineering IL-2 selectivity

The model is used to explore receptor-binding changes that favor signaling in target CD8⁺ T and NK cell populations relative to non-target populations, including Tregs and Th17 cells.

- [`exec_slopiness_target.py`](exec_slopiness_target.py) — evaluates the sensitivity of the target-versus-non-target response to IL-2 receptor-binding parameters.
- [`pyabc_mutant_search.py`](pyabc_mutant_search.py) — uses ABC-SMC to search for receptor-binding parameter combinations that enhance target-cell selectivity.
- [`IL2_mutants.ipynb`](IL2_mutants.ipynb) — analyzes and plots experimentally tested IL-2 mutants, including variants designed to enhance IL-2Rγ binding.
- [`Plot_ABC_SMC.ipynb`](Plot_ABC_SMC.ipynb) — additionally visualizes inferred parameter distributions from the mutant search.

**Typical inputs:** fitted wild-type parameters, calibrated RNA-to-protein models, and melanoma cell-type expression profiles.

---

## Data availability

- **Analysis and simulation code:** https://github.com/Quim98/IL2_model_publication
- **Processed experimental data, model initial conditions, inference outputs, and simulations:**  Zenodo, 
- External published datasets used in individual analyses are described and cited in the manuscript and within the corresponding analysis notebooks.

## Issues

Please report missing files, installation problems, or code-related issues using the [GitHub issue tracker](https://github.com/Quim98/IL2_model_publication/issues). Include the operating system, relevant package versions, filename being run, and complete error message.

## Authors



## Citation

Please cite the associated IL-2 study when using this code. The definitive manuscript citation and DOI will be added here once they are available.
