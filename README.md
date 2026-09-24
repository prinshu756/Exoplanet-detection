# AI-Enabled Exoplanet Detection from Noisy Astronomical Light Curves

An end-to-end research pipeline that ingests TESS light curves, detects transit
signals, and classifies each signal into one of four astrophysical classes using
a PyTorch neural network — served through a Flask web application.

> Built as the artifact for a research paper. Every stage is reproducible from
> the notebooks in `notebooks/`.

---

## Table of Contents

- [What the system does](#what-the-system-does)
- [Classification classes](#classification-classes)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the web application](#running-the-web-application)
- [Using the application](#using-the-application)
- [API reference](#api-reference)
- [Re-running the research pipeline](#re-running-the-research-pipeline)
- [Retraining the classifier](#retraining-the-classifier)
- [How transit search is isolated (important)](#how-transit-search-is-isolated-important)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Current results](#current-results)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## What the system does

1. **Ingests** light curves and catalogue data from the NASA Exoplanet Archive
   and TESS.
2. **Cleans** each light curve (outlier removal, detrending, gap handling).
3. **Detects** transits with **Transit Least Squares (TLS)** plus BLS, scoring
   candidates by SDE, transit count and fit quality.
4. **Extracts** 50 features per light curve — statistical, temporal,
   frequency-domain and transit-derived.
5. **Classifies** each candidate into one of four classes with a trained
   PyTorch MLP.
6. **Explains** predictions with SHAP, feature importance and permutation tests.
7. **Estimates** physical parameters (radius, equilibrium temperature) via
   transit modelling and MCMC uncertainty quantification.
8. **Serves** everything through a Flask UI where users can upload their own
   light curve and get a classification, TLS periodogram statistics, and
   raw / processed / phase-folded plots.

---

## Classification classes

| Label | Class               | Description                                            |
| ----- | ------------------- | ------------------------------------------------------ |
| `0`   | Planet              | Transit consistent with a planetary body                |
| `1`   | Eclipsing Binary    | Deep, often secondary-eclipse / V-shaped eclipses       |
| `2`   | Background Blend    | Shallow, low-SDE events, often near-crowded contaminants |
| `3`   | Stellar Variability | No coherent transit; periodic or irregular variability |

---

## Tech stack

| Layer       | Technology                                                                     |
| ----------- | ------------------------------------------------------------------------------ |
| Language    | Python 3.12                                                                    |
| Web         | Flask 3, Jinja2 templates, vanilla JS, CSS (light/dark theme)                  |
| DL / ML     | PyTorch, scikit-learn, joblib                                                  |
| Data        | pandas, NumPy, SciPy, PyArrow (Parquet), Astropy (FITS)                        |
| Astronomy   | `transitleastsquares` (TLS), `batman` (transit models), `emcee` (MCMC), Lightkurve |
| Explainability | SHAP, permutation importance, partial-dependence-style diagnostics            |
| Notebook    | Jupyter (the reproducible research pipeline)                                   |
| Acceleration | CUDA via PyTorch when available (automatic CPU fallback)                      |

---

## Repository structure

```
Exoplanet-detection/
├── frontend/                     # Flask web application
│   ├── app.py                    # Routes + server entrypoint
│   ├── predictions.py            # Model inference, features, TLS orchestration
│   ├── tls_worker.py             # Lightweight TLS subprocess (memory isolation)
│   ├── feature_utils.py          # 50-feature extraction shared with training
│   ├── train_model.py            # Training / cross-validation entrypoint
│   ├── requirements.txt
│   ├── templates/                # index, candidates, candidate_detail, predictions
│   └── static/css/style.css
│
├── notebooks/                    # The research pipeline, in order
│   ├── 01_Data_Ingestion.ipynb
│   ├── 02_Light_Curve_Acquisition_Pipeline.ipynb
│   ├── 03_Light_Curve_Preprocessing.ipynb
│   ├── 04_Transit_Detection&Candidate_Detetion.ipynb
│   ├── 05_Feature_Engineering.ipynb
│   ├── 06_AI_Classification_Engine.ipynb
│   ├── 07_Explainable_AI_Scientific_Representation.ipynb
│   ├── 08_Transit_Parameter_Estimation.ipynb
│   └── Data_Quality_Analysis.ipynb
│
├── data/
│   ├── raw/                      # Downloaded catalogues + light curves
│   │   ├── catalogs/             # NASA Exoplanet Archive (TOI/CTOI/confirmed)
│   │   ├── lightcurves/          # TIC_<id>_Sector_<n>.parquet
│   │   └── manual_labels.csv     # Curated 4-class ground truth
│   └── processed/
│       ├── lightcurves/          # Cleaned light curves (one Parquet per star)
│       ├── candidates/           # Candidate catalogue + selection summary
│       ├── features/             # Feature metadata
│       ├── explainability/       # Feature table, SHAP values + explainer
│       └── master_catalog.*      # Unified catalogue (6,028 rows)
│
├── models/                       # Trained artifacts (required to run the app)
│   ├── best_model.pth            # PyTorch state dict
│   ├── scaler.pkl                # StandardScaler
│   ├── label_encoder.pkl
│   ├── feature_columns.json      # Feature ordering
│   └── random_forest.pkl         # Baseline model
│
├── outputs/
│   ├── candidate_parameters/     # Per-candidate transit parameters
│   ├── transit_models/           # Best-fit transit models
│   ├── mcmc/                     # Posterior samples + statistics
│   └── reports/                  # research_summary.md
│
├── reports/                      # Quality, classification, feature reports
├── figures/                      # Generated figures
├── logs/                         # Per-notebook execution logs
└── README.md
```

> `data/` is git-ignored (large binaries). `models/` is committed so the web app
> runs out of the box.

---

## Prerequisites

- **Python 3.12** (recommended — `transitleastsquares` / `numba` are most stable here)
- `pip` and `venv`
- ~2 GB free RAM minimum for the app; **8 GB+ recommended** (see
  [TLS isolation](#how-transit-search-is-isolated-important))
- Optional: NVIDIA GPU with a CUDA-enabled PyTorch build (training only;
  inference falls back to CPU automatically)

---

## Setup

```bash
git clone https://github.com/prinshu756/Exoplanet-detection.git
cd Exoplanet-detection

python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install the frontend dependencies:

```bash
pip install --upgrade pip
pip install -r frontend/requirements.txt
```

The notebook pipeline has additional dependencies (install these only if you plan
to re-run the research stages):

```bash
pip install lightkurve astroquery wotan plotly batman-package emcee shap numba cupy-cuda12x
```

Verify the installation:

```bash
python -c "import torch, flask, transitleastsquares; print('deps OK')"
```

### Required files

The app expects these to exist relative to the repository root:

| Path                                    | Purpose                        | In repo? |
| --------------------------------------- | ------------------------------ | -------- |
| `models/best_model.pth`                 | Neural network weights         | ✅        |
| `models/scaler.pkl`                     | Feature standardisation       | ✅        |
| `models/feature_columns.json`           | Feature ordering               | ✅        |
| `outputs/candidate_parameters/candidate_parameters.json` | Candidate list shown in the UI | ✅ |
| `data/processed/lightcurves/*.parquet`  | Used to classify all candidates | ⚠️ clone does not include `data/` |

If `data/processed/lightcurves/` is missing, the app falls back to
`data/processed/explainability/feature_table.csv`; if that is missing too, the
candidate list still loads but the AI prediction counts will be empty. Upload
analysis (`/api/analyze`) works regardless — it does not need the dataset.

---

## Running the web application

```bash
cd frontend
python app.py
```

Expected output:

```
 * 23 candidates, 17 with predictions
 * Summary: {'total': 23, 'nn_counts': {'Eclipsing Binary': 12, 'Background Blend': 11}}
 * Running on http://localhost:5000
```

Open <http://localhost:5000>.

To use a different port:

```bash
# macOS / Linux
PORT=8080 python app.py

# Windows PowerShell
$env:PORT=8080; python app.py
```

The server binds to `0.0.0.0`, so it is also reachable from other devices on
your network (e.g. `http://192.168.x.x:5000` from a phone on the same Wi-Fi).

> This is Flask's development server. For production use a real WSGI server
> (gunicorn on Linux, waitress on Windows) behind a reverse proxy.

---

## Using the application

### Analyse a light curve

1. Go to the **Analyse** tab.
2. Drop a `.parquet`, `.fits` or `.csv` file onto the upload zone (or click to browse).
3. The file must contain **`time`** and **`flux`** columns.
4. The app runs a TLS period search over **0.5 – 50 days**, extracts 50 features,
   and returns:
   - the predicted class and confidence,
   - the full probability distribution,
   - TLS period and SDE,
   - raw, processed and phase-folded plots.

A typical run takes 1–3 minutes depending on the number of data points.

### Browse candidates

- **Candidates** — table of every detected candidate.
- Click a candidate for its transit parameters, quality score and prediction.
- Toggle the dark/light theme with the sun/moon button in the header.

---

## API reference

| Method | Endpoint             | Description                                       |
| ------ | -------------------- | ------------------------------------------------- |
| `GET`  | `/`                  | Analysis page (with embedded candidate data)      |
| `GET`  | `/candidates`        | Candidate list page                               |
| `GET`  | `/candidates/<tic>`  | Candidate detail page (404 if unknown)            |
| `GET`  | `/api/candidates`    | All candidates as JSON                            |
| `GET`  | `/api/predictions`   | Model predictions keyed by TIC ID                 |
| `POST` | `/api/analyze`       | Analyse an uploaded light curve                   |

### `POST /api/analyze`

Multipart form upload under the field name `file`. Max size 200 MB.

```bash
curl -F "file=@data/processed/lightcurves/TIC_152476657_Sector_4.parquet" \
     http://localhost:5000/api/analyze
```

**Response (200)**

```json
{
  "n_points": 17117,
  "time_span_days": 27.41,
  "mean_flux": 1.0,
  "std_flux": 0.0031,
  "snr": 12.7,
  "tls_period": 7.4443,
  "tls_sde": 9.81,
  "prediction": "Planet",
  "probabilities": {
    "Planet": 0.71,
    "Eclipsing Binary": 0.14,
    "Background Blend": 0.10,
    "Stellar Variability": 0.05
  },
  "confidence": 0.71,
  "plots": {
    "raw": "<base64 PNG>",
    "processed": "<base64 PNG>",
    "folded": "<base64 PNG>"
  }
}
```

**Error responses**

| Status | Body                                      | Cause                                  |
| ------ | ----------------------------------------- | -------------------------------------- |
| `400`  | `{"error": "No file"}`                    | Missing `file` field                  |
| `400`  | `{"error": "Use .parquet, .fits, or .csv"}` | Unsupported extension                |
| `400`  | `{"error": "Missing columns: [...]"}`     | No `time` / `flux` columns             |
| `500`  | `{"error": "Analysis error: ..."}`        | Internal failure                       |

> If TLS fails the app degrades gracefully: `tls_error` is populated, the
> transit-derived features fall back to defaults, and the neural-network
> classification is still returned.

---

## Re-running the research pipeline

The notebooks are numbered and depend on the output of the previous one. Run
them in this order from the `notebooks/` directory (or from the repository root
after updating `ROOT_DIR` in each notebook).

| #    | Notebook                            | Produces                                                                    |
| ---- | ----------------------------------- | --------------------------------------------------------------------------- |
| 01   | `01_Data_Ingestion.ipynb`           | Catalogue tables in `data/raw/catalogs/`, project scaffolding                |
| 02   | `02_Light_Curve_Acquisition_Pipeline.ipynb` | TESS light curves in `data/raw/lightcurves/`, `download_report.csv`   |
| —    | `Data_Quality_Analysis.ipynb`       | Quality metrics, outlier and preprocessing reports in `reports/`            |
| 03   | `03_Light_Curve_Preprocessing.ipynb`| Cleaned light curves in `data/processed/lightcurves/`                       |
| 04   | `04_Transit_Detection&Candidate_Detetion.ipynb` | TLS/BLS search → `data/processed/candidates/`, candidate list    |
| 05   | `05_Feature_Engineering.ipynb`      | 50-feature table in `data/processed/features/`                             |
| 06   | `06_AI_Classification_Engine.ipynb` | Trains the NN + Random Forest baseline, writes `models/`                    |
| 07   | `07_Explainable_AI_Scientific_Representation.ipynb` | SHAP values, feature importance, explainability figures |
| 08   | `08_Transit_Parameter_Estimation.ipynb` | Transit fits, MCMC posteriors, `research_summary.md`                    |

Notebook parameters currently in use for the candidate search:

```python
MIN_PERIOD   = 0.5    # days
MAX_PERIOD   = 20.0   # days
MIN_DURATION = 0.04   # days (~1 hour)
MAX_DURATION = 0.5    # days (~12 hours)
MIN_SDE      = 7.0
MIN_TRANSITS = 2
```

Each notebook writes a log to `logs/notebook_XX.log`.

---

## Retraining the classifier

`frontend/train_model.py` reproduces the training stage independently of
notebook 06:

```bash
cd frontend
python train_model.py
```

It:

- loads the labelled light curves from `data/raw/manual_labels.csv`,
- generates physics-informed synthetic augmentations (real transits, eclipsing
  binaries, blends, stellar variability) plus noise augmentation,
- runs 5-fold stratified cross-validation,
- trains the final model with AdamW + cosine warm restarts + early stopping,
- writes `models/best_model.pth`, `models/scaler.pkl`,
  `models/label_encoder.pkl` and `models/feature_columns.json`.

Key hyper-parameters: `lr=3e-4`, `weight_decay=5e-3`, batch size 64, label
smoothing 0.1, up to 100 (CV) / 150 (final) epochs with early stopping
(patience 20 / 25). Seeds are fixed at 42 for reproducibility.

The feature extractor (`feature_utils.py`) is shared between training and
inference, so the 50-feature ordering is guaranteed to match.

---

## How transit search is isolated (important)

TLS (`transitleastsquares`) opens a `multiprocessing.Pool`. On Windows, every
spawned worker **re-imports the main module**. If TLS is imported directly in
`app.py`, each worker would reload the whole application — including PyTorch
and its CUDA libraries — which previously caused:

```
OSError: [WinError 1455] The paging file is too small for this operation to complete
LLVM ERROR: Unable to allocate memory for common symbols
MemoryError
```

**The fix:** TLS runs in a separate subprocess whose entrypoint
(`frontend/tls_worker.py`) imports only NumPy and TLS. TLS's own worker pool
therefore re-imports a tiny script instead of the full application, and the
Flask process is never duplicated.

The parent communicates with the worker through a temporary `.npz` (time and
flux) and receives a JSON result (period, SDE, duration, depth, epoch, SNR, and
the folded model curve used for plotting). The subprocess is given a 15-minute
timeout and its temp directory is always cleaned up.

---

## Configuration

| Variable      | Default | Description                                              |
| ------------- | ------- | -------------------------------------------------------- |
| `PORT`        | `5000`  | Port the Flask server binds to                            |
| `TLS_THREADS` | `8`     | Worker processes TLS uses for the period search           |

```bash
# Windows PowerShell — reduce memory pressure on small machines
$env:TLS_THREADS=4
python app.py
```

Lower `TLS_THREADS` if you have less than 8 GB of free RAM; raise it (up to your
core count) for faster searches on larger machines.

---

## Troubleshooting

**`The paging file is too small` / `WinError 1455` / `LLVM ERROR: Unable to allocate memory`**
System is out of commit memory — almost always caused by too many TLS workers.
Set `TLS_THREADS=4` (or `2`) and restart. See the section above.

**`No file` / `Missing columns` on upload**
The uploaded file must have columns named exactly `time` and `flux`. FITS files
are read from HDU 1.

**`0 candidates` or empty prediction counts**
`data/` is git-ignored. Run notebooks 02–06, or ensure
`data/processed/lightcurves/` and
`data/processed/explainability/feature_table.csv` exist.

**`Models not loaded`**
`models/*.pth` / `models/*.pkl` are missing. Either run `train_model.py` or
restore the files.

**Port already in use**
Change the port: `PORT=5001 python app.py`.

**Analysis is very slow**
TLS searches a 0.5–50 day grid. Long light curves (tens of thousands of points)
take longer. `TLS_THREADS` controls the trade-off between speed and memory.

**`transitleastsquares` fails to install on Python 3.13+**
Its `numba` dependency often lags new Python releases. Use **Python 3.12**.

---

## Current results

Pipeline output as of the latest run (see `outputs/reports/research_summary.md`):

- **89** light curves processed → **23** transit candidates retained
- 1 candidate flagged **CONFIRMED**, 22 **PROMISING**
- Median orbital period **3.45 d** (range 0.61–13.28 d)
- Median planet radius **0.68 R⊕** (range 0.10–2.01 R⊕)
- Median equilibrium temperature **1037 K**
- MCMC uncertainty estimated for 3 candidates
- Median reduced χ² **1.003**

Classification performance (18-sample held-out test set in
`reports/classification_report.txt`):

| Model          | Accuracy | Macro F1 |
| -------------- | -------- | -------- |
| Neural network | 0.56     | 0.56     |
| Random Forest  | 0.61     | 0.62     |

> These figures come from a small curated label set and are reported for
> transparency. See [Limitations](#limitations).

---

## Limitations

- The curated ground-truth set is small (18 labelled targets), so held-out
  metrics have wide confidence intervals and the Random Forest baseline still
  edges out the neural network. Augmentation is used to broaden training
  coverage, but the test set itself is not augmented.
- Reported candidate parameters are preliminary: median R² of the transit fits is
  low (~0.01), so physical parameters (radius, temperature) should be treated as
  indicative rather than publication-grade.
- MCMC uncertainty quantification was only run for 3 of the 23 candidates.
- The UI uses Flask's development server; it is not hardened for public
  deployment (no auth, no rate limiting, no CSRF protection).
- Candidate search is limited to periods between 0.5 and 20 days in the research
  pipeline, and 0.5–50 days in the live app.

---

## Citation

This project accompanies a research paper on AI-enabled exoplanet detection from
noisy TESS light curves. If you use it, please cite the repository and the paper
once published.

```bibtex
@misc{exoplanet_detection_ai,
  title  = {AI-Enabled Exoplanet Detection from Noisy Astronomical Light Curves},
  author = {{Prinshu}},
  year   = {2026},
  note   = {GitHub repository: prinshu756/Exoplanet-detection}
}
```

Data derived from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
and the TESS mission.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
