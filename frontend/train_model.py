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
from sklearn.feature_selection import VarianceThreshold
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = torch.log_softmax(pred, dim=1)
        return -(smooth_labels * log_probs).sum(dim=1).mean()


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


def load_real_data():
    labels = pd.read_csv(os.path.join(DATA, "raw", "manual_labels.csv"))
    catalog = pd.read_parquet(os.path.join(CANDIDATES, "candidate_catalog.parquet"))
    candidate_map = catalog.set_index("tic_id").to_dict(orient="index")

    lc_index = {}
    for fname in os.listdir(LIGHTCURVES):
        if not fname.endswith(".parquet"):
            continue
        parts = fname.replace(".parquet", "").split("_")
        try:
            tic_idx = parts.index("TIC") + 1
            tic = int(parts[tic_idx])
            lc_index.setdefault(tic, []).append(fname)
        except (ValueError, IndexError):
            continue

    records = []
    for _, row in labels.iterrows():
        tic = row["tic_id"]
        tic_files = sorted(lc_index.get(tic, []))
        if not tic_files:
            continue
        df = pd.read_parquet(os.path.join(LIGHTCURVES, tic_files[0]))
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


def generate_synthetic_from_template(template_df, label, n_per_template=5):
    t = template_df["time"].values.astype(np.float64)
    f = template_df["flux"].values.astype(np.float64)

    time_span = t[-1] - t[0]
    records = []

    for _ in range(n_per_template):
        f_synth = f.copy()
        period = np.random.uniform(0.6, min(time_span * 0.8, 15.0))
        duration = period * np.random.uniform(0.01, 0.05)

        if label == 0:
            depth = np.random.uniform(0.0005, 0.02)
            duration = period * np.random.uniform(0.01, 0.03)
        elif label == 1:
            depth = np.random.uniform(0.005, 0.12)
            duration = period * np.random.uniform(0.02, 0.07)
        elif label == 2:
            depth = np.random.uniform(0.0002, 0.008)
            duration = period * np.random.uniform(0.008, 0.04)
        elif label == 3:
            n_modes = np.random.randint(2, 5)
            variability = np.zeros_like(t)
            for mi in range(n_modes):
                amp = np.random.uniform(0.002, 0.04)
                p_var = np.random.uniform(0.3, min(time_span * 0.5, 10.0))
                phase_var = np.random.uniform(0, p_var)
                variability += amp * np.sin(2 * np.pi * (t - t[0]) / p_var + phase_var)
            f_synth += variability
            depth = 0
            duration = period * 0.01

        if label in (0, 1, 2):
            epoch = t[0] + np.random.uniform(time_span * 0.1, time_span * 0.9)
            phases = (t - epoch + period / 2) % period - period / 2
            transit_bottom = duration * 0.7 if label == 0 else duration * 0.3
            ing = (duration - transit_bottom) / 2
            egr = duration - transit_bottom - ing

            f_synth = f_synth.copy()
            in_transit = np.abs(phases) < duration / 2
            if in_transit.any():
                transit_phases = phases[in_transit]
                tp = np.abs(transit_phases)
                depth_profile = np.ones_like(tp)

                if label in (0, 2):
                    transit_bottom = duration * 0.75 if label == 0 else duration * 0.3
                    ing = (duration - transit_bottom) / 2
                    egr = duration - transit_bottom - ing
                    flat_start = transit_bottom / 2
                    depth_profile[tp < flat_start] = 0
                    mask_ing = (tp >= flat_start) & (tp < flat_start + ing)
                    mask_egr = (tp >= flat_start + ing)
                    if mask_ing.any():
                        depth_profile[mask_ing] = (tp[mask_ing] - flat_start) / ing
                    if mask_egr.any():
                        depth_profile[mask_egr] = 1 - (tp[mask_egr] - flat_start - ing) / egr
                else:
                    depth_profile = tp / (duration / 2)
                    depth_profile = np.sqrt(depth_profile)

                    if np.random.random() < 0.4:
                        sec_shift = period / 2
                        sec_phases = (t - epoch + sec_shift + period / 2) % period - period / 2
                        in_sec = np.abs(sec_phases) < duration / 2
                        if in_sec.any():
                            sec_tp = np.abs(sec_phases[in_sec])
                            sec_profile = np.sqrt(sec_tp / (duration / 2))
                            sec_depth = depth * np.random.uniform(0.3, 0.7)
                            f_synth[in_sec] -= sec_depth * sec_profile

                if label == 2:
                    depth_profile = depth_profile ** 1.5
                    if np.random.random() < 0.3:
                        lumpy = 1 + 0.2 * np.sin(tp * np.pi * np.random.uniform(3, 8))
                        depth_profile = depth_profile * lumpy

                depth_profile = np.clip(depth_profile, 0, 1)
                f_synth[in_transit] -= depth * depth_profile

        noise_level = np.random.uniform(0.08, 0.35)
        if label == 2:
            noise_level = np.random.uniform(0.2, 0.6)
            hf_noise = np.random.normal(0, np.std(f_synth) * 0.1, len(f_synth))
            f_synth += hf_noise
            if np.random.random() < 0.3:
                f_synth += np.linspace(0, np.random.uniform(-0.003, 0.003), len(t))
        f_synth = add_noise_augmentation(f_synth, noise_level)

        if label == 0:
            sde = np.random.uniform(6, 20)
            snr = np.random.uniform(8, 30)
        elif label == 1:
            sde = np.random.uniform(4, 14)
            snr = np.random.uniform(3, 14)
        elif label == 2:
            sde = np.random.uniform(1.5, 9)
            snr = np.random.uniform(1.0, 7)
        else:
            sde = np.random.uniform(1.5, 9)
            snr = np.random.uniform(0.5, 7)

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


def generate_augmented_data(real_records, multiplier=30):
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
    n_templates_total = max(1, len(template_dfs) // 6)
    for label_val, name in LABEL_MAP.items():
        n_real = label_counts.get(label_val, 0)
        needed = target_per_class - n_real
        if needed <= 0:
            continue
        n_templates = min(n_templates_total, len(template_dfs))
        n_per = max(3, needed // n_templates)
        for lf in list(template_dfs.keys())[:n_templates]:
            df = template_dfs[lf]
            syn = generate_synthetic_from_template(df, label_val, n_per_template=n_per)
            synthetic.extend(syn)

    all_records = []
    for r in real_records:
        all_records.append({"features": r["features"], "label": r["label"]})
        for _ in range(5):
            noisy_feats = r["features"] + np.random.normal(0, 0.005, size=r["features"].shape)
            all_records.append({"features": noisy_feats, "label": r["label"]})
    all_records.extend(synthetic)

    X = np.stack([r["features"] for r in all_records])
    y = np.array([r["label"] for r in all_records])
    return X, y, len(real_records)


def remove_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    kept_mask = selector.get_support()
    return X_reduced, kept_mask


def train():
    print("Loading real data...")
    real = load_real_data()
    print(f"  Real samples: {len(real)}")
    label_dist = {}
    for r in real:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"  Class distribution: {label_dist}")

    print("Generating augmented data...")
    X, y, n_real = generate_augmented_data(real, multiplier=20)
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

    class_counts = np.bincount(y_enc)
    class_weights = torch.FloatTensor(len(class_counts) / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_enc)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        model = TransitClassifier(X_scaled.shape[1], n_classes)
        model.to(DEVICE)
        criterion = LabelSmoothCrossEntropy(smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        train_t = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_t = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_t, batch_size=64, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_t, batch_size=64, drop_last=False)

        best_val_acc = 0
        patience = 0
        for epoch in range(100):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(torch.FloatTensor(X_val).to(DEVICE))
                val_pred = val_out.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                val_loss = criterion(val_out, torch.LongTensor(y_val).to(DEVICE)).item()

            scheduler.step(epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                if val_acc > best_score:
                    best_score = val_acc
                    best_model = model.state_dict().copy()
            else:
                patience += 1
                if patience >= 20:
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
    model.to(DEVICE)
    criterion = LabelSmoothCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    train_t = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_t = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_t, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_t, batch_size=64, drop_last=False)

    best_test_acc = 0
    patience = 0

    for epoch in range(150):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_out = model(torch.FloatTensor(X_test).to(DEVICE))
            test_pred = test_out.argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="weighted")
            test_loss = criterion(test_out, torch.LongTensor(y_test).to(DEVICE)).item()

        scheduler.step(epoch)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODELS, "best_model.pth"))
        else:
            patience += 1
            if patience >= 25:
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: test acc = {test_acc:.4f}, f1 = {test_f1:.4f}")

    model.load_state_dict(torch.load(os.path.join(MODELS, "best_model.pth"), map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        final_pred = model(torch.FloatTensor(X_test).to(DEVICE)).argmax(dim=1).cpu().numpy()
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
