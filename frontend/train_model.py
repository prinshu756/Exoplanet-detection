import json, os, pickle, warnings, gc
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from feature_utils import extract_features_from_lc_df, features_to_array, FEATURE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA, "processed")
LIGHTCURVES = os.path.join(PROCESSED, "lightcurves")
CANDIDATES = os.path.join(PROCESSED, "candidates")

LABEL_MAP = {0: "Planet", 1: "Eclipsing Binary", 2: "Background Blend", 3: "Stellar Variability"}

torch.manual_seed(42)
np.random.seed(42)


class TransitClassifier(nn.Module):
    def __init__(self, input_dim=50, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_real_data():
    labels = pd.read_csv(os.path.join(DATA, "raw", "manual_labels.csv"))
    catalog = pd.read_parquet(os.path.join(CANDIDATES, "candidate_catalog.parquet"))
    candidate_map = catalog.set_index("tic_id").to_dict(orient="index")

    records = []
    for _, row in labels.iterrows():
        tic = row["tic_id"]
        lc_files = sorted(
            f for f in os.listdir(LIGHTCURVES)
            if f.startswith(f"TIC_{tic}_") and f.endswith(".parquet")
        )
        if not lc_files:
            continue
        df = pd.read_parquet(os.path.join(LIGHTCURVES, lc_files[0]))
        cinfo = candidate_map.get(tic, {})
        transit_info = {}
        for col in ["period", "duration", "depth", "epoch", "sde", "transit_snr"]:
            val = cinfo.get(col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                transit_info[col] = val

        feats = extract_features_from_lc_df(df, transit_info or None)
        arr = features_to_array(feats)
        records.append({"tic_id": tic, "features": arr, "label": row["label"], "class_name": row["class_name"]})
    return records


def add_noise_augmentation(f, noise_level=0.15):
    noise = np.random.normal(0, np.std(f) * noise_level, len(f))
    return f + noise


def add_time_jitter(t, jitter_level=0.001):
    jitter = np.random.normal(0, jitter_level, len(t))
    return t + jitter


def generate_synthetic_from_template(template_df, label, n_per_template=5):
    t = template_df["time"].values.astype(np.float64)
    f = template_df["flux"].values.astype(np.float64)

    time_span = t[-1] - t[0]
    records = []

    for _ in range(n_per_template):
        f_synth = f.copy()
        period = np.random.uniform(0.6, min(time_span * 0.8, 15.0))
        depth = np.random.uniform(0.0001, 0.05)
        duration = period * np.random.uniform(0.01, 0.05)

        if label == 0:
            depth = np.random.uniform(0.0001, 0.02)
            duration = period * np.random.uniform(0.01, 0.03)
            ing = duration * 0.08
            egr = duration * 0.08
        elif label == 1:
            depth = np.random.uniform(0.005, 0.10)
            duration = period * np.random.uniform(0.02, 0.07)
            ing = duration * 0.05
            egr = duration * 0.05
        elif label == 2:
            depth = np.random.uniform(0.0002, 0.008)
            duration = period * np.random.uniform(0.008, 0.035)
            ing = duration * 0.25
            egr = duration * 0.15
        elif label == 3:
            n_modes = np.random.randint(1, 4)
            variability = np.zeros_like(t)
            for mi in range(n_modes):
                amp = np.random.uniform(0.003, 0.04)
                p_var = np.random.uniform(0.3, min(time_span * 0.5, 10.0))
                phase_var = np.random.uniform(0, p_var)
                variability += amp * np.sin(2 * np.pi * (t - t[0]) / p_var + phase_var)
            f_synth += variability
            depth = 0
            duration = period * 0.01
            ing = egr = 0

        if label in (0, 1, 2):
            epoch = t[0] + np.random.uniform(time_span * 0.1, time_span * 0.9)
            transit_bottom = duration - ing - egr
            phases = (t - epoch + period / 2) % period - period / 2

            f_synth = f_synth.copy()
            in_transit = np.abs(phases) < duration / 2
            if in_transit.any():
                transit_phases = phases[in_transit]
                tp = np.abs(transit_phases)
                depth_profile = np.ones_like(tp)
                depth_profile[tp < transit_bottom / 2] = 0
                mask_ing = (tp >= transit_bottom / 2) & (tp < transit_bottom / 2 + ing)
                mask_egr = (tp >= transit_bottom / 2 + ing)
                if mask_ing.any():
                    depth_profile[mask_ing] = (tp[mask_ing] - transit_bottom / 2) / ing
                if mask_egr.any():
                    depth_profile[mask_egr] = 1 - (tp[mask_egr] - transit_bottom / 2 - ing) / egr

                if label == 1:
                    depth_profile = np.sqrt(depth_profile)
                if label == 2:
                    depth_profile = depth_profile ** 1.5

                depth_profile = np.clip(depth_profile, 0, 1)
                f_synth[in_transit] -= depth * depth_profile

        noise_level = np.random.uniform(0.1, 0.4)
        if label == 2:
            noise_level = np.random.uniform(0.2, 0.6)
            hf_noise = np.random.normal(0, np.std(f_synth) * 0.1, len(f_synth))
            f_synth += hf_noise
        f_synth = add_noise_augmentation(f_synth, noise_level)

        if label == 0:
            sde = np.random.uniform(6, 18)
            snr = np.random.uniform(8, 25)
        elif label == 1:
            sde = np.random.uniform(4, 12)
            snr = np.random.uniform(3, 12)
        elif label == 2:
            sde = np.random.uniform(2, 8)
            snr = np.random.uniform(1.5, 6)
        else:
            sde = np.random.uniform(2, 8)
            snr = np.random.uniform(1, 6)

        transit_info = {"period": period, "duration": duration,
                        "depth": float(np.median(f) - np.min(f_synth)),
                        "epoch": float(t[0] + time_span / 2),
                        "sde": float(sde),
                        "transit_snr": float(snr)}

        feats = extract_features_from_lc_df(
            pd.DataFrame({"time": t, "flux": f_synth}), transit_info
        )
        arr = features_to_array(feats)
        records.append({"features": arr, "label": label})
    return records


def generate_augmented_data(real_records, multiplier=20):
    lc_files = sorted(os.listdir(LIGHTCURVES))
    template_dfs = {}
    for lf in lc_files:
        df = pd.read_parquet(os.path.join(LIGHTCURVES, lf))
        template_dfs[lf] = df

    label_counts = {}
    for r in real_records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    max_count = max(label_counts.values()) if label_counts else 1
    target_per_class = max_count * multiplier

    synthetic = []
    for label_val, name in LABEL_MAP.items():
        n_real = label_counts.get(label_val, 0)
        needed = target_per_class - n_real
        if needed <= 0:
            continue
        n_templates = max(1, len(template_dfs) // 4)
        n_per = max(1, needed // n_templates)
        for lf in list(template_dfs.keys())[:n_templates]:
            df = template_dfs[lf]
            syn = generate_synthetic_from_template(df, label_val, n_per_template=n_per)
            synthetic.extend(syn)

    all_records = []
    for r in real_records:
        all_records.append({"features": r["features"], "label": r["label"]})
    all_records.extend(synthetic)

    X = np.stack([r["features"] for r in all_records])
    y = np.array([r["label"] for r in all_records])
    return X, y, len(real_records)


def train():
    print("Loading real data...")
    real = load_real_data()
    print(f"  Real samples: {len(real)}")
    label_dist = {}
    for r in real:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"  Class distribution: {label_dist}")

    print("Generating augmented data...")
    X, y, n_real = generate_augmented_data(real, multiplier=30)
    print(f"  Total samples: {len(X)} (real: {n_real}, synthetic: {len(X) - n_real})")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_enc)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        model = TransitClassifier(X_scaled.shape[1], n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8
        )

        train_t = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_t = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_t, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_t, batch_size=32, drop_last=False)

        best_val_acc = 0
        patience = 0
        for epoch in range(100):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(torch.FloatTensor(X_val))
                val_pred = val_out.argmax(dim=1).numpy()
                val_acc = accuracy_score(y_val, val_pred)
                val_loss = criterion(val_out, torch.LongTensor(y_val)).item()

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                if val_acc > best_score:
                    best_score = val_acc
                    best_model = model.state_dict().copy()
            else:
                patience += 1
                if patience >= 15:
                    break

        fold_scores.append(best_val_acc)
        print(f"  Fold {fold+1}: best val acc = {best_val_acc:.4f}")

    print(f"\nCross-val accuracy: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    train_final_model(X_scaled, y_enc, le, scaler)


def train_final_model(X_scaled, y_enc, le, scaler):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    model = TransitClassifier(X_scaled.shape[1], len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    train_t = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_t = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_t, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_t, batch_size=32, drop_last=False)

    best_test_acc = 0
    patience = 0

    for epoch in range(150):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_out = model(torch.FloatTensor(X_test))
            test_pred = test_out.argmax(dim=1).numpy()
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="weighted")
            test_loss = criterion(test_out, torch.LongTensor(y_test)).item()

        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODELS, "best_model.pth"))
        else:
            patience += 1
            if patience >= 20:
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: test acc = {test_acc:.4f}, f1 = {test_f1:.4f}")

    model.load_state_dict(torch.load(os.path.join(MODELS, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        final_pred = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
    print(f"\nFinal test accuracy: {best_test_acc:.4f}")
    print(f"Final test F1: {f1_score(y_test, final_pred, average='weighted'):.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_test, final_pred, target_names=[str(c) for c in le.classes_]))

    pickle.dump(scaler, open(os.path.join(MODELS, "scaler.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(MODELS, "label_encoder.pkl"), "wb"))
    json.dump(FEATURE_NAMES, open(os.path.join(MODELS, "feature_columns.json"), "w"))

    print(f"\nModels saved to {MODELS}/")


if __name__ == "__main__":
    train()
