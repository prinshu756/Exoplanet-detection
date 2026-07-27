import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "mean_flux", "median_flux", "std_flux", "variance",
    "minimum_flux", "maximum_flux", "dynamic_range",
    "rms", "mad",
    "skewness", "kurtosis",
    "percentile_5", "percentile_25", "percentile_75", "percentile_95", "iqr",
    "flux_entropy", "zero_crossings",
    "observation_duration_days", "num_observations",
    "median_cadence_days", "mean_cadence_days", "std_cadence_days",
    "cadence_minutes", "largest_gap_days", "mean_gap_days",
    "gap_fraction", "sampling_density", "time_span_hours",
    "dominant_frequency", "dominant_period", "dominant_power",
    "second_frequency", "second_power",
    "third_frequency", "third_power",
    "mean_spectral_power", "max_spectral_power", "spectral_std",
    "spectral_entropy", "spectral_energy", "harmonic_ratio",
    "power_concentration", "frequency_variance",
    "transit_period", "transit_duration", "transit_depth",
    "transit_epoch", "transit_sde", "transit_transit_snr",
]


def compute_statistical_features(flux):
    f = flux[~np.isnan(flux)]
    if len(f) == 0:
        return {}

    median_f = np.median(f)
    return {
        "mean_flux": float(np.mean(f)),
        "median_flux": float(median_f),
        "std_flux": float(np.std(f)),
        "variance": float(np.var(f)),
        "minimum_flux": float(np.min(f)),
        "maximum_flux": float(np.max(f)),
        "dynamic_range": float(np.max(f) - np.min(f)),
        "rms": float(np.sqrt(np.mean(f ** 2))),
        "mad": float(np.median(np.abs(f - median_f))),
        "skewness": float(pd.Series(f).skew()),
        "kurtosis": float(pd.Series(f).kurtosis()),
        "percentile_5": float(np.percentile(f, 5)),
        "percentile_25": float(np.percentile(f, 25)),
        "percentile_75": float(np.percentile(f, 75)),
        "percentile_95": float(np.percentile(f, 95)),
        "iqr": float(np.percentile(f, 75) - np.percentile(f, 25)),
        "flux_entropy": float(-np.sum((f / f.sum()) * np.log(f / f.sum() + 1e-12))),
        "zero_crossings": int(np.sum(np.diff(np.sign(f - median_f)) != 0)),
    }


def compute_temporal_features(time):
    t = time[~np.isnan(time)]
    if len(t) < 2:
        return {}

    diffs = np.diff(t)
    time_span = t[-1] - t[0]
    median_cad = np.median(diffs)
    return {
        "observation_duration_days": float(time_span),
        "num_observations": int(len(t)),
        "median_cadence_days": float(median_cad),
        "mean_cadence_days": float(np.mean(diffs)),
        "std_cadence_days": float(np.std(diffs)),
        "cadence_minutes": float(median_cad * 1440),
        "largest_gap_days": float(np.max(diffs)),
        "mean_gap_days": float(np.mean(diffs)),
        "gap_fraction": float(np.sum(diffs[diffs > 1.5 * median_cad]) / time_span) if time_span > 0 else 0,
        "sampling_density": float(len(t) / time_span) if time_span > 0 else 0,
        "time_span_hours": float(time_span * 24),
    }


def compute_frequency_features(flux, time):
    f = flux[~np.isnan(flux)]
    if len(f) < 10:
        return {}

    n = len(f)
    window = np.hanning(n)
    f_detrend = f - np.mean(f)
    fft_vals = np.fft.rfft(f_detrend * window)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=np.nanmedian(np.diff(time)))
    freqs = freqs[1:]
    power = power[1:]

    if len(power) == 0:
        return {}

    idx_sorted = np.argsort(power)[::-1]
    top_idx = idx_sorted[:3]
    top_idx = [t for t in top_idx if t < len(power)]
    for _ in range(3 - len(top_idx)):
        top_idx.append(0)

    p_sum = power.sum()
    p_norm = power / p_sum if p_sum > 0 else power
    entropy = -np.sum(p_norm * np.log(p_norm + 1e-12))

    harm_ratio = float(power[top_idx[0]] / power[top_idx[1]]) if (
        len(top_idx) > 1 and power[top_idx[1]] > 0
    ) else 0

    n_top = max(5, len(power) // 10)
    pow_conc = float(np.sum(power[:n_top]) / p_sum) if p_sum > 0 else 0

    return {
        "dominant_frequency": float(freqs[top_idx[0]]),
        "dominant_period": float(1.0 / freqs[top_idx[0]]) if freqs[top_idx[0]] > 0 else 0,
        "dominant_power": float(power[top_idx[0]]),
        "second_frequency": float(freqs[top_idx[1]]) if len(top_idx) > 1 and top_idx[1] < len(freqs) else 0,
        "second_power": float(power[top_idx[1]]) if len(top_idx) > 1 else 0,
        "third_frequency": float(freqs[top_idx[2]]) if len(top_idx) > 2 and top_idx[2] < len(freqs) else 0,
        "third_power": float(power[top_idx[2]]) if len(top_idx) > 2 else 0,
        "mean_spectral_power": float(np.mean(power)),
        "max_spectral_power": float(np.max(power)),
        "spectral_std": float(np.std(power)),
        "spectral_entropy": float(entropy),
        "spectral_energy": float(np.sum(power)),
        "harmonic_ratio": harm_ratio,
        "power_concentration": pow_conc,
        "frequency_variance": float(np.var(power)),
    }


def extract_features(flux, time, transit_info=None):
    features = {}
    features.update(compute_statistical_features(flux))
    features.update(compute_temporal_features(time))
    features.update(compute_frequency_features(flux, time))
    if transit_info is not None:
        for k, v in transit_info.items():
            features[f"transit_{k}"] = float(v) if v is not None else 0.0
    return features


def features_to_array(feature_dict, feature_names=None):
    if feature_names is None:
        feature_names = FEATURE_NAMES
    arr = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        arr[i] = feature_dict.get(name, 0.0)
    return arr


def array_to_dict(arr, feature_names=None):
    if feature_names is None:
        feature_names = FEATURE_NAMES
    return {name: float(arr[i]) for i, name in enumerate(feature_names)}


def extract_features_from_lc_df(df, transit_info=None):
    flux = df["flux"].values
    time = df["time"].values
    return extract_features(flux, time, transit_info)
