<p align="center">
  <img src="DHDS_logo.png" alt="DHDS — Digital Health Data Science Lab, Korea University College of Medicine" width="560">
</p>

<h1 align="center">KNHANES + NHANES Research Automation Platform</h1>

<p align="center">
  A reproducible, survey-weighted analysis platform that turns national health-examination microdata
  into publication-ready epidemiologic, temporal-trend, and machine-learning reports.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/R-survey-informational" alt="R survey">
  <img src="https://img.shields.io/badge/app-Streamlit-red" alt="streamlit">
  <img src="https://img.shields.io/badge/stats-deterministic-2e7d32" alt="deterministic">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

<p align="center">
  <b>Digital Health Data Science Lab (DHDS)</b><br>
  Department of Biomedical Informatics, Korea University College of Medicine<br>
  Contact: <a href="mailto:seogsongjeong@korea.ac.kr">seogsongjeong@korea.ac.kr</a>
</p>

---

## Overview

This platform standardizes the full analytic pipeline for the Korea National Health and Nutrition
Examination Survey (KNHANES) and the U.S. National Health and Nutrition Examination Survey (NHANES).
An investigator selects exposures, outcomes, and operational definitions in a web interface; all
estimation is performed with complex-survey weights, and the platform emits formatted Microsoft Word
(`.docx`) reports for three study designs. A large language model, when available, drafts the report
prose in English from the computed values and never recomputes any statistic.

**Design principle.** Every number is a deterministic survey-weighted estimate produced by the R
`survey` package. The language model is confined to variable mapping and narrative writing. Covariate
sets are constructed to avoid over-adjustment, and machine-learning feature sets are screened for
outcome-definition leakage. All findings are hypothesis-generating and require external validation.

## Key features

- **Interactive analysis.** Any variable can serve as an exposure or an outcome. Continuous outcomes
  are modeled with survey linear regression (beta per standard deviation); binary outcomes with survey
  logistic regression (odds ratios). False discovery rate controls multiplicity.
- **Epidemiologic report.** Baseline Table 1 with standardized mean differences, crude and adjusted
  odds ratios, subgroup analysis with interaction tests, a causal-inference panel
  (PSM, IPTW, AIPW, G-computation, TMLE), Love plot, variance inflation factors, restricted cubic
  spline nonlinearity, and E-values.
- **Trend report.** Yearly prevalence, counts, age-sex direct-standardized rates, linear projection
  with prediction intervals, joinpoint segment annual percent change with bootstrap intervals, and a
  negative-binomial forecast.
- **Machine-learning report.** Sixteen classifiers with cross-validated tuning, a full metric panel,
  ROC and precision-recall curves, threshold analysis, calibration, SHAP attributions, and partial
  dependence, with automatic exclusion of leakage features.
- **Trajectory view.** Age trajectories of the variables that define a chosen outcome.

## Operational definitions (publication standard)

Diabetes (ADA), hypertension (JNC), metabolic syndrome (harmonized NCEP ATP III, three of five),
dyslipidemia (multi-criterion), hepatic steatosis (NAFLD-LFS or HSI for KNHANES; controlled
attenuation parameter for NHANES), and smoking and alcohol grouped as never versus past or current.
Covariate selection automatically excludes the exposure, the outcome, the outcome's definitional
components, and the anthropometric cluster to prevent over-adjustment.

## Repository layout

| Path | Purpose |
| --- | --- |
| `factory_app.py` | Streamlit web application (entry point) |
| `factory_core.py` | Loaders, operational definitions, unified variables, auto covariates, survey engine, trajectory, LLM |
| `engine.R` | Table 1 and associations (linear beta / logistic OR by outcome type) |
| `epi_report.py`, `epi.R`, `epi_adv.R` | Epidemiologic report and causal-inference panel |
| `trend_report.py`, `trend.R` | Trend report (standardized rates, joinpoint APC, NB forecast) |
| `ml_report.py` | Machine-learning report (16 models, SHAP, PDP, leakage screening) |
| `docs/index.html` | Project landing page (GitHub Pages) |
| `docs/sample_report.html` | Example static output |

## Important: GitHub Pages cannot run the tool

GitHub Pages is static hosting and cannot execute Python or R or read local SAS/XPT files. The
interactive application therefore runs locally, on the machine where the data resides; data never
leaves that machine. GitHub Pages hosts only the landing page and a sample static report. Streamlit
Community Cloud is unsuitable because it requires uploading private, licensed data, and browser-based
Python (stlite/Pyodide) cannot run R `survey`.

## Installation

```bash
pip install -r requirements.txt
# R engine (Debian/Ubuntu)
sudo apt-get install -y r-base r-cran-survey r-cran-jsonlite r-cran-mass
# optional: local LLM for narrative prose (deterministic fallback otherwise)
ollama pull llama3.3:70b && ollama serve
```

On Windows, install R from CRAN, run `install.packages(c("survey","jsonlite","MASS"))`, and ensure
`Rscript` is on the system PATH.

## Data layout (kept local)

```
data/KNHANES/  hn08_all.sas7bdat  hn08_dxa.sas7bdat  hn09_*  hn10_*  hn11_* ...
data/NHANES/   demo_j  alq_j  bmx_j  biopro_j  glu_j  trigly_j  tchol_j  lux_j  bpx_j  smq_j  dxx_j ...
```

Both `.sas7bdat` and `.XPT` are supported. KNHANES `all` and `dxa` files are merged on the respondent
key; NHANES modules are merged on `SEQN`. Paths are configurable in the sidebar and searched
recursively. Data files are excluded from version control by `.gitignore`.

## Running

```bash
streamlit run factory_app.py
```

The application opens at `http://localhost:8501`. On a firewalled server, forward the port over SSH
(`ssh -L 8501:localhost:8501 user@server`) and open the same address locally.

## Machine-learning models

LogisticRegression, ElasticNet-Logistic, LDA, GaussianNB, KNN, DecisionTree, RandomForest, ExtraTrees,
AdaBoost, GradientBoosting, HistGradientBoosting, SVM-RBF, MLP, XGBoost, LightGBM, and CatBoost.
SVM-RBF and MLP are deselected by default for speed and remain available.

## Citation

If this platform contributes to your work, please cite it:

```bibtex
@software{dhds_knhanes_nhanes_platform,
  title   = {KNHANES + NHANES Research Automation Platform},
  author  = {Digital Health Data Science Lab, Korea University College of Medicine},
  year    = {2025},
  note    = {Department of Biomedical Informatics. Contact: seogsongjeong@korea.ac.kr},
  url     = {https://github.com/kumcdhds-a11y/Health-Analysis-Network-Scientist}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE). In brief, the software is provided
"as is", without warranty of any kind; you may use, modify, and distribute it, including for
commercial purposes, provided the copyright notice and permission notice are retained.

## Contact

Digital Health Data Science Lab (DHDS), Department of Biomedical Informatics,
Korea University College of Medicine — seogsongjeong@korea.ac.kr

---

<p align="center"><sub>All estimates are deterministic survey-weighted results. Findings are hypothesis-generating and require external validation.</sub></p>
