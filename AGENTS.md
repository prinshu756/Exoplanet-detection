# Project Knowledge

## Pipeline Overview
- **Goal**: Classify TESS light curves into 4 classes: Planet, Eclipsing Binary, Background Blend, Stellar Variability
- **Data**: 89 labeled samples with 50 features each
- **Model**: PyTorch NN (50→96→48→24→4) with BatchNorm, Dropout(0.35/0.25/0.15), Label Smoothing, CosineAnnealing

## Key Files

### `frontend/feature_utils.py`
Single source of truth for feature extraction. Contains:
- `compute_statistical_features(flux)` — 16 features (removed redundant variance/rms, added autocorr_lag1, flux_range_ratio, rolling_std_ratio)
- `compute_temporal_features(flux, time)` — 11 features using pd.Series.rolling for fast std
- `compute_frequency_features(flux, time)` — 17 features (+spectral_skew/kurtosis)
- `extract_features(flux, time, transit_info)` — combines all + optional transit features
- `features_to_array(feature_dict, feature_names)` — converts dict to numpy array (default: 50 FEATURE_NAMES order)
- `extract_features_from_lc_df(df, transit_info)` — convenience wrapper
- `FEATURE_NAMES` — 50 feature names in canonical order

### `frontend/predictions.py`
Uses `feature_utils` for extraction. Loads `best_model.pth` (50→96→48→24→4), `scaler.pkl`, `feature_columns.json`. Functions:
- `predict_candidates()` — loads candidate_parameters.json with TRANSIT_COLUMN_MAP for column name normalization
- `analyze_light_curve(file_path, ext)` — upload handler, runs TLS then extracts features and classifies
- `classification_summary()` — counts by class

### `frontend/train_model.py`
- Multiplies data ~25× via synthetic transit injection into real TESS noise templates
- Uses V-shaped (EB/BB) and U-shaped (planet) transit profiles with limb darkening
- BB generation: asymmetric V-shapes, lumpy profiles, micro-variability, higher noise
- EB generation: includes optional secondary eclipses
- Stellar Var: multi-frequency sine + random flares
- 5-fold StratifiedKFold cross-validation
- Arch: 50→96→48→24→4, AdamW (lr=3e-4, wd=5e-3), CosineAnnealingWarmRestarts, LabelSmoothing(0.1), patience 20/25
- Saves to `models/`: best_model.pth, scaler.pkl, label_encoder.pkl, feature_columns.json

### `frontend/app.py`
Flask app with routes: `/`, `/candidates`, `/candidates/<tic_id>`, `/api/candidates`, `/api/predictions`, `/api/analyze`

### `models/`
Contains: best_model.pth (NN weights 50→96→48→24→4), scaler.pkl (StandardScaler), label_encoder.pkl (LabelEncoder), feature_columns.json (50 feature names), random_forest.pkl (legacy)

## Data Files
- `data/raw/manual_labels.csv` — 89 ground-truth labels (tic_id, label 0-3, class_name)
- `data/processed/explainability/feature_table.csv` — 89 rows × 53 cols — legacy
- `data/processed/candidates/candidate_catalog.parquet` — 89-row catalog used by training pipeline
- `data/processed/lightcurves/TIC_{ID}_Sector_{S}.parquet` — individual light curve parquets
- `outputs/candidate_parameters/candidate_parameteres.json` — 23 records (17 unique TIC IDs) with transit-fit params (uses different column names like orbital_period_days)

## Critical Fixes Applied
1. **Feature mismatch**: `feature_utils.py` created as single source of truth; predictions.py and train_model.py both use it
2. **Data augmentation**: Synthetic transit injection into real TESS noise; ~2200 synthetic samples
3. **NN architecture**: 50→96→48→24→4 (9.6K params) with higher dropout (0.35/0.25/0.15)
4. **Training improvements**: Label smoothing (0.1), CosineAnnealingWarmRestarts, AdamW (3e-4, 5e-3)
5. **BB generation**: Asymmetric V-shaped transits, lumpy profiles, trend injection, higher noise
6. **EB generation**: Secondary eclipses, sqrt depth profiles
7. **Temporal features**: rolling_std via pd.Series.rolling (200× speedup), autocorr_lag1, flux_range_ratio
8. **Spectral features**: spectral_skew, spectral_kurtosis added
9. **Column mapping**: TRANSIT_COLUMN_MAP in predictions.py for candidate_parameters.json column name normalization
10. **Real data loading**: Indexed by TIC ID to avoid repeated os.listdir (89× → 1×)
11. **Frontend dark mode**: CSS variables with `.dark` class, stored in localStorage

## Current Model Performance
- **Cross-val accuracy**: 91.2%
- **Final test accuracy**: 93.6%
- **Final test F1**: 0.936
- Per class (F1): Planet=0.96, EB=0.90, BB=0.91, Stellar Var=0.97

## Known Issues
- Class 2 (Background Blend) F1 improved from 0.53 to 0.91 but BB signals remain hardest
- 6 of 23 candidate records lack matching light curves (TIC ID matching or missing downloads)
- `candidate_parameters.json` has 23 records but only 17 unique TIC IDs (duplicate entries)
- `notebooks/06_AI_Classification_Engine.ipynb` was updated with smaller model but still lacks full augmentation pipeline
