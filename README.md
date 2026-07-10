# KNHANES + NHANES Research Automation Platform

Select variables and operational definitions in a web UI; survey-weighted analyses run internally and
**epidemiologic, trend, and machine-learning reports are generated as Word (.docx)**. Streamlit app.

## It does NOT "run" on github.io (GitHub Pages) — important
GitHub Pages is **static hosting** (HTML/JS/CSS only). It cannot execute Python/R/Streamlit or read local
SAS/XPT files. So the interactive tool itself cannot run on github.io. For a research tool whose data is local:

- **Code lives in the GitHub repository** (this repo).
- **Run it locally/on your server where the data is**: `streamlit run factory_app.py`. Data never leaves your machine.
- **github.io (`docs/`)** hosts a landing/usage page and a **sample static report** (`docs/sample_report.html`) that
  shows what the tool produces. (Settings → Pages → Branch `main`, Folder `/docs`)

Note: Streamlit Community Cloud requires uploading data (unsuitable for private/licensed KNHANES), and browser
execution (stlite/Pyodide) cannot run R survey, so it does not fit this tool.

## Files
- `factory_app.py` — Streamlit web app (entry point)
- `factory_core.py` — loaders, operational definitions, unified variables, auto covariates, survey-weighted analysis, trajectory, LLM
- `engine.R` — Table 1 + associations (continuous outcome linear beta / binary outcome logistic OR)
- `epi_report.py` + `epi.R` + `epi_adv.R` + `epi_surv.R` — epidemiologic report
  (Table 1-6, crude/adjusted OR, subgroup + interaction, PSM · IPTW · AIPW · G-computation · TMLE, Love plot, VIF, RCS, E-value)
  and survival analysis (survey-weighted Cox `svycoxph`, KM curves, incidence)
- `trend_report.py` + `trend.R` — trend report (prevalence, counts, age-sex standardized rate, projection, joinpoint segment APC, NB forecast)
- `ml_report.py` — ML report (16 models incl. XGBoost/LightGBM/CatBoost/HistGB, tuning, full metrics, ROC/PR, threshold, calibration, SHAP, PDP; automatic leakage blocking)
- `docs/index.html` — github.io landing; `docs/sample_report.html` — sample output

## Install
```bash
pip install -r requirements.txt
sudo apt-get install -y r-base r-cran-survey r-cran-jsonlite r-cran-rms r-cran-mass r-cran-sandwich
ollama pull llama3.3:70b && ollama serve   # optional; without it, report prose falls back to deterministic text
```

## Data layout (stays on your machine)
- KNHANES: `./data/KNHANES/hn08_all.sas7bdat`, `./data/KNHANES/hn08_dxa.sas7bdat` (per year: all/dxa)
- NHANES: `./data/NHANES/demo_j`, `alq_j`, `bmx_j` ...  — **both `.sas7bdat` and `.XPT` are supported**
Paths are configurable in the sidebar; subfolders are searched recursively.

## Run
```bash
streamlit run factory_app.py
# Firewalled remote server (e.g., hospital network): SSH tunnel, then browser http://localhost:8501
ssh -L 8501:localhost:8501 user@server
```

## Tabs
1. Interactive analysis · 2. Trajectory · 3. Epidemiologic report · 4. Trend report · 5. ML report (each produces Word)

## 16 ML models
LogisticRegression, ElasticNet-Logistic, LDA, GaussianNB, KNN, DecisionTree, RandomForest, ExtraTrees,
AdaBoost, GradientBoosting, HistGradientBoosting, SVM-RBF, MLP, XGBoost, LightGBM, CatBoost
(SVM-RBF and MLP excluded by default because they are slow; selectable.)

## Operational definitions (publication standard) · principles
Diabetes ADA · Hypertension JNC · Metabolic syndrome harmonized NCEP (3 of 5) · Dyslipidemia multi-criterion ·
Steatosis NAFLD-LFS/HSI/CAP · Smoking/alcohol never vs ever. Covariates auto-exclude the exposure, outcome,
outcome determinants, and the anthropometric cluster (over-adjustment prevention). ML auto-excludes outcome
definitional components (leakage). All statistics are deterministic survey-weighted estimates (R survey); the LLM
only writes report prose in English and never recomputes numbers. All findings are hypothesis-generating.

## Survival / incidence (cohort variables required)
Survival death/MACE and incidence require follow-up time and event variables (e.g., NHANES Linked Mortality File,
or a cohort/claims dataset). KNHANES/NHANES exam files do not contain these. `build_survival_report(...)` and the
Cox/KM/incidence code activate once those columns are provided.
