import json
import sys

import numpy as np


def run_tls(t, f, period_min, period_max, use_threads):
    from transitleastsquares import transitleastsquares
    model = transitleastsquares(t, f)
    return model.power(period_min=period_min, period_max=period_max,
                       use_threads=use_threads)


def _serialize(res, t0):
    return {
        "period": float(res.period),
        "sde": float(res.SDE),
        "fap": float(getattr(res, "FAP", 0.0)),
        "duration": float(getattr(res, "duration", 0.0)),
        "depth": float(getattr(res, "depth", 0.0)),
        "epoch": float(getattr(res, "epoch", t0)),
        "snr": float(getattr(res, "snr", 0.0)),
        "transit_count": int(getattr(res, "transit_count", 0)),
        "model_folded_phase": np.asarray(
            getattr(res, "model_folded_phase", []), dtype=np.float64
        ).tolist(),
        "model_lightcurve": np.asarray(
            getattr(res, "model_folded_model", []), dtype=np.float64
        ).tolist(),
    }


if __name__ == "__main__":
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    t = np.asarray(data["time"], dtype=np.float64)
    f = np.asarray(data["flux"], dtype=np.float64)
    period_min = float(data["period_min"])
    period_max = float(data["period_max"])
    use_threads = int(data["use_threads"])
    try:
        res = run_tls(t, f, period_min, period_max, use_threads)
        result = _serialize(res, float(t[0]))
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)