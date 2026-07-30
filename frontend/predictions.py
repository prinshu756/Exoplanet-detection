import base64, io, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_utils import extract_features_from_lc_df, features_to_array, FEATURE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUTPUTS = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")
LIGHTCURVES = os.path.join(DATA, "processed", "lightcurves")
LABEL_MAP = {0: "Planet", 1: "Eclipsing Binary", 2: "Background Blend", 3: "Stellar Variability"}

TRANSIT_COLUMN_MAP = {
    "orbital_period_days": "period",
    "transit_epoch_btjd": "epoch",
    "transit_depth": "depth",
    "transit_duration_hours": "duration",
    "sde": "sde",
    "transit_snr": "transit_snr",
    "period": "period",
    "duration": "duration",
    "epoch": "epoch",
}


class TransitClassifier(nn.Module):
    def __init__(self, input_dim=50, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        return self.net(x)


_models = None


def get_models():
    global _models
    if _models is not None:
        return _models

    nn_model = TransitClassifier()
    nn_model.load_state_dict(
        torch.load(os.path.join(MODELS, "best_model.pth"),
                   map_location="cpu")
    )
    nn_model.eval()

    scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))

    with open(os.path.join(MODELS, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    _models = {
        "nn": nn_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    return _models


def predict_candidates():
    m = get_models()

    candidates = pd.read_json(
        os.path.join(OUTPUTS, "candidate_parameters", "candidate_parameters.json")
    )

    tic_ids = candidates["tic_id"].unique().tolist()
    lc_dir = LIGHTCURVES
    if not os.path.isdir(lc_dir):
        feat_table = pd.read_csv(
            os.path.join(DATA, "processed", "explainability", "feature_table.csv")
        )
        can_tic = set(int(t) for t in tic_ids)
        feat_for_can = feat_table[feat_table["tic_id"].isin(can_tic)].copy()
        old_cols = [c for c in feat_table.columns if c not in ("tic_id", "label", "class_name")]
        if feat_for_can.empty or not old_cols:
            return candidates.to_dict(orient="records"), {}
        X = feat_for_can[old_cols].values
        given_tic_ids = feat_for_can["tic_id"].values.astype(int)
        if X.shape[1] != 50:
            return candidates.to_dict(orient="records"), {}
        df_pred, predictions = _predict_from_features(X, given_tic_ids, m)
    else:
        lc_files = sorted(os.listdir(lc_dir))
        features_list = []
        valid_tic_ids = []
        for lf in lc_files:
            parts = lf.replace(".parquet", "").split("_")
            try:
                tic_idx = parts.index("TIC") + 1
                tic = int(parts[tic_idx])
            except (ValueError, IndexError):
                continue
            if tic not in tic_ids:
                continue
            df = pd.read_parquet(os.path.join(lc_dir, lf))
            row = candidates[candidates["tic_id"] == tic].iloc[0] if tic in candidates["tic_id"].values else None
            transit_info = None
            if row is not None:
                raw = {k: v for k, v in row.to_dict().items()
                       if isinstance(v, (int, float)) and not np.isnan(v)}
                transit_info = {}
                for src_key, tgt_key in TRANSIT_COLUMN_MAP.items():
                    if src_key in raw:
                        transit_info[tgt_key] = raw[src_key]
            feats = extract_features_from_lc_df(df, transit_info)
            arr = features_to_array(feats, m["feature_cols"])
            features_list.append(arr)
            valid_tic_ids.append(tic)

        if not features_list:
            return candidates.to_dict(orient="records"), {}

        X = np.stack(features_list)
        df_pred, predictions = _predict_from_features(X, np.array(valid_tic_ids), m)

    augmented = []
    for _, row in candidates.iterrows():
        rec = row.to_dict()
        tic = int(rec["tic_id"])
        pred = predictions.get(tic, {})
        rec["nn_prediction"] = pred.get("prediction", "\u2014")
        rec["nn_confidence"] = max(pred.get("probabilities", {}).values(), default=0)
        augmented.append(rec)

    return augmented, predictions


def _predict_from_features(X, tic_ids, m):
    X_scaled = m["scaler"].transform(X)
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = m["nn"](X_tensor)
        proba = torch.softmax(logits, dim=1).numpy()
    preds = proba.argmax(axis=1)

    predictions = {}
    for i, tic in enumerate(tic_ids):
        label = int(preds[i])
        predictions[int(tic)] = {
            "prediction": LABEL_MAP.get(label, "Unknown"),
            "probabilities": {
                LABEL_MAP.get(j, f"Class_{j}"): float(proba[i][j])
                for j in range(4)
            },
        }
    return [], predictions


def analyze_light_curve(file_path, ext):
    if ext == "parquet":
        lc = pd.read_parquet(file_path)
    elif ext == "fits":
        from astropy.io import fits
        hdul = fits.open(file_path)
        lc = pd.DataFrame(hdul[1].data)
        hdul.close()
    else:
        lc = pd.read_csv(file_path)

    required = ["time", "flux"]
    missing = [c for c in required if c not in lc.columns]
    if missing:
        return {"error": f"Missing columns: {missing}. Need: time, flux"}

    t_raw = lc["time"].values.astype(np.float64)
    f_raw = lc["flux"].values.astype(np.float64)
    f = f_raw / np.median(f_raw)
    lc["flux"] = f

    result = {
        "n_points": int(len(t_raw)),
        "time_span_days": float(t_raw.max() - t_raw.min()),
        "mean_flux": float(np.mean(f)),
        "std_flux": float(np.std(f)),
        "snr": float(np.std(f) / (np.mean(np.abs(np.diff(f))) + 1e-10)),
    }

    transit_info = None
    tls_res = None
    try:
        from transitleastsquares import transitleastsquares
        model = transitleastsquares(t_raw, f)
        tls_res = model.power(period_min=0.5, period_max=50)
        result["tls_period"] = float(tls_res.period)
        result["tls_sde"] = float(tls_res.SDE)
        transit_info = {
            "period": float(tls_res.period),
            "duration": float(getattr(tls_res, "duration", 0.0)),
            "depth": float(getattr(tls_res, "depth", 0.0)),
            "epoch": float(getattr(tls_res, "epoch", t_raw[0])),
            "sde": float(getattr(tls_res, "SDE", 0.0)),
            "transit_snr": float(getattr(tls_res, "snr", 0.0)),
        }
    except Exception as e:
        result["tls_error"] = str(e)
        transit_info = {
            "period": 1.0, "duration": 0.0, "depth": 0.0,
            "epoch": float(t_raw[0]), "sde": 0.0, "transit_snr": 0.0,
        }

    m = get_models()
    feats = extract_features_from_lc_df(lc, transit_info)
    arr = features_to_array(feats, m["feature_cols"]).reshape(1, -1)

    try:
        X_scaled = m["scaler"].transform(arr)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            logits = m["nn"](X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(proba.argmax())
        result["prediction"] = LABEL_MAP.get(pred, "Unknown")
        result["probabilities"] = {
            LABEL_MAP.get(j, f"Class_{j}"): float(proba[j])
            for j in range(4)
        }
        result["confidence"] = float(proba.max())
    except Exception as e:
        result["classification_error"] = str(e)

    result["plots"] = _generate_plots(
        t_raw, f_raw, t_raw, f, period=transit_info.get("period"),
        epoch=transit_info.get("epoch"), tls_res=tls_res
    )

    return result


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _generate_plots(t_raw, f_raw, t_norm, f_norm, period=None, epoch=None, tls_res=None):
    plots = {}

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(t_raw, f_raw, s=0.5, c="#437bb1", alpha=0.6, rasterized=True)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("Raw Light Curve")
    ax.ticklabel_format(useOffset=False)
    fig.tight_layout()
    plots["raw"] = _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(t_norm, f_norm, s=0.5, c="#059669", alpha=0.5, rasterized=True)
    window = max(5, len(t_norm) // 50)
    smoothed = pd.Series(f_norm).rolling(window=window, center=True).mean().values
    ax.plot(t_norm, smoothed, c="#d97706", lw=1, alpha=0.9, label="Smoothed")
    ax.axhline(1.0, c="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Processed Light Curve")
    ax.ticklabel_format(useOffset=False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots["processed"] = _fig_to_b64(fig)

    if period and period > 0 and tls_res is not None:
        t_fold = ((t_norm - (epoch or t_norm[0])) % period) / period
        model_lc = getattr(tls_res, "model_lightcurve", None)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.scatter(t_fold, f_norm, s=0.5, c="#7c3aed", alpha=0.4, rasterized=True)
        if model_lc is not None and hasattr(tls_res, "model_folded_phase"):
            ax.plot(tls_res.model_folded_phase, model_lc, c="#dc2626", lw=2, label="TLS Model")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"Phase-folded (P = {period:.4f} d)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plots["folded"] = _fig_to_b64(fig)

    return plots


def classification_summary(candidates=None, predictions=None):
    counts = {}
    total = 0
    for c in candidates or []:
        total += 1
        nn = c.get("nn_prediction", "\u2014")
        counts[nn] = counts.get(nn, 0) + 1
    return {"total": total, "nn_counts": counts}
