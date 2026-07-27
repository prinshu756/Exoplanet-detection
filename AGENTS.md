# Project Knowledge

## Pipeline Overview
- **Goal**: Classify TESS light curves into 4 classes: Planet, Eclipsing Binary, Background Blend, Stellar Variability
- **Data**: 89 labeled samples with 50 features each
- **Model**: PyTorch NN (50→64→32→4) with BatchNorm, Dropout, AdamW, ReduceLROnPlateau

## Key Files

### `frontend/feature_utils.py`
Single source of truth for feature extraction. Contains:
- `compute_statistical_features(flux)` — 18 features (mean, median, std, var, min, max, range, rms, mad, skew, kurtosis, percentiles, iqr, entropy, zero_crossings)
- `compute_temporal_features(time)` — 11 features (duration, N, cadence stats, gap_fraction, sampling_density, time_span_hours)
- `compute_frequency_features(flux, time)` — 15 features using Hanning window + `np.fft.rfft` (dominant freq/power, harmonics, spectral entropy/energy, harmonic_ratio, power_concentration, freq_variance)
- `extract_features(flux, time, transit_info)` — combines all + optional transit features
- `features_to_array(feature_dict, feature_names)` — converts dict to numpy array (default: 50 FEATURE_NAMES order)
- `extract_features_from_lc_df(df, transit_info)` — convenience wrapper
- `FEATURE_NAMES` — 50 feature names in canonical order

### `frontend/predictions.py`
Uses `feature_utils` for extraction. Loads `best_model.pth` (50→64→32→4), `scaler.pkl`, `feature_columns.json`. Functions:
- `predict_candidates()` — loads candidate_parameters.json, matches light curves by TIC ID from `data/processed/lightcurves/TIC_{ID}_Sector_{S}.parquet`, falls back to legacy `feature_table.csv` if lightcurves dir missing
- `analyze_light_curve(file_path, ext)` — upload handler, runs TLS then extracts features and classifies
- `classification_summary()` — counts by class

### `frontend/train_model.py`
- Multiplies data 30× via synthetic transit injection into real TESS noise templates
- 5-fold StratifiedKFold cross-validation
- Saves to `models/`: best_model.pth, scaler.pkl, label_encoder.pkl, feature_columns.json
- Config arch: 50→64→32→4, AdamW (lr=0.001, wd=1e-3), ReduceLROnPlateau, early stopping (patience 15/20), gradient clipping

### `frontend/app.py`
Flask app with routes: `/`, `/candidates`, `/candidates/<tic_id>`, `/api/candidates`, `/api/predictions`, `/api/analyze`

### `models/`
Contains: best_model.pth (NN weights), scaler.pkl (StandardScaler), label_encoder.pkl (LabelEncoder), feature_columns.json (50 feature names), random_forest.pkl (legacy)

## Data Files
- `data/raw/manual_labels.csv` — 89 ground-truth labels (tic_id, label 0-3, class_name)
- `data/processed/explainability/feature_table.csv` — 89 rows × 53 cols (50 features + tic_id + label + class_name) — legacy, old feature extraction
- `data/processed/candidates/candidate_catalog.parquet` — 89-row catalog used by training pipeline
- `data/processed/lightcurves/TIC_{ID}_Sector_{S}.parquet` — individual light curve parquets (not all 89 TICs overlapping with candidate_parameters.json)
- `outputs/candidate_parameters/candidate_parameteres.json` — 23 records (17 unique TIC IDs) with transit-fit params

## Critical Fixes Applied
1. **Feature mismatch**: `feature_utils.py` created as single source of truth; predictions.py and train_model.py both use it
2. **Data augmentation**: Synthetic transit injection into real TESS noise; 30× multiplier → ~2700 samples
3. **NN architecture**: Simplified from 50→256→128→64→32→4 (62K params) to 50→64→32→4 (3.2K params) — reduces overfitting on 89 samples
4. **Training improvements**: AdamW with weight decay, ReduceLROnPlateau, early stopping, gradient clipping, K-fold CV
5. **Frontend dark mode**: CSS variables with `.dark` class, stored in localStorage

## Known Issues
- Class 2 (Background Blend) F1=0.53 — BB signals fundamentally hard to distinguish; synthetic generation needs improvement (use real BB template features)
- 6 of 23 candidate records lack matching light curves in the parquet files (TIC ID matching issue or missing downloads)
- `candidate_parameters.json` has 23 records but only 17 unique TIC IDs (duplicate entries for same TIC)
