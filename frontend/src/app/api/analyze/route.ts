import { NextRequest, NextResponse } from "next/server"
import { execSync } from "child_process"
import { writeFile, mkdir, unlink } from "fs/promises"
import { join } from "path"
import { randomUUID } from "crypto"

export async function POST(req: NextRequest) {
  const tmpDir = join(process.cwd(), "..", "data", "uploads")
  await mkdir(tmpDir, { recursive: true })

  try {
    const formData = await req.formData()
    const file = formData.get("file") as File
    if (!file) return NextResponse.json({ error: "No file provided" }, { status: 400 })

    const ext = file.name.split(".").pop()?.toLowerCase()
    if (!ext || !["parquet", "fits", "csv"].includes(ext)) {
      return NextResponse.json({ error: "Unsupported format. Use .parquet, .fits, or .csv" }, { status: 400 })
    }

    const id = randomUUID().slice(0, 8)
    const tmpPath = join(tmpDir, `upload_${id}.${ext}`)
    const buf = Buffer.from(await file.arrayBuffer())
    await writeFile(tmpPath, buf)

    const scriptPath = join(tmpDir, `analyze_${id}.py`)
    const root = join(process.cwd(), "..").replace(/\\/g, "/")

    const script = `
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "${root}")
import numpy as np
import pandas as pd

ext = "${ext}"
path = "${tmpPath.replace(/\\/g, '/')}"
if ext == "parquet":
    lc = pd.read_parquet(path)
elif ext == "fits":
    from astropy.io import fits
    hdul = fits.open(path)
    lc = pd.DataFrame(hdul[1].data)
    hdul.close()
else:
    lc = pd.read_csv(path)

required = ["time", "flux"]
missing = [c for c in required if c not in lc.columns]
if missing:
    print(json.dumps({"error": f"Missing columns: {missing}. Need: time, flux"}))
    sys.exit(0)

t = lc["time"].values.astype(np.float64)
f = lc["flux"].values.astype(np.float64)
f = f / np.median(f)

result = {
    "n_points": int(len(t)),
    "time_span_days": float(t.max() - t.min()),
    "mean_flux": float(np.mean(f)),
    "std_flux": float(np.std(f)),
    "snr": float(np.std(f) / (np.mean(np.abs(np.diff(f))) + 1e-10)),
}

try:
    from transitleastsquares import transitleastsquares
    model = transitleastsquares(t, f)
    tls_res = model.power(period_min=0.5, period_max=50)
    result["tls_period"] = float(tls_res.period)
    result["tls_sde"] = float(tls_res.SDE)
    result["tls_snr"] = float(max(tls_res.power))
except Exception:
    pass

try:
    import joblib
    rf = joblib.load("${root}/models/random_forest.pkl")
    scaler = joblib.load("${root}/models/scaler.pkl")
    le = joblib.load("${root}/models/label_encoder.pkl")

    period = result.get("tls_period", 1.0)
    depth = float(1 - np.min(f))
    dur = float(len(t[t < np.percentile(t, 25)]) / len(t) * (t.max() - t.min()))
    feats = np.array([[period, depth, dur, result["snr"], float(np.std(f)),
        float(np.mean(f)), float(np.median(f)), float(np.percentile(f, 25)),
        float(np.percentile(f, 75)), float(np.abs(np.fft.fft(f - np.mean(f))[1]))]])
    if feats.shape[1] < 50:
        feats = np.pad(feats, ((0,0), (0, 50 - feats.shape[1])), mode="constant")
    feats_scaled = scaler.transform(feats)
    pred = rf.predict(feats_scaled)[0]
    proba = rf.predict_proba(feats_scaled)[0].tolist()
    result["class_prediction"] = str(le.inverse_transform([pred])[0])
    result["class_probabilities"] = {str(le.inverse_transform([i])[0]): float(p) for i, p in enumerate(proba)}
except Exception as e:
    result["classification_error"] = str(e)

print(json.dumps(result))
`

    await writeFile(scriptPath, script)
    const output = execSync(`python "${scriptPath.replace(/"/g, '\\"')}"`, {
      encoding: "utf-8",
      timeout: 120000,
      maxBuffer: 10 * 1024 * 1024,
    }).trim()

    // Cleanup
    unlink(tmpPath).catch(() => {})
    unlink(scriptPath).catch(() => {})

    const parsed = JSON.parse(output)
    return NextResponse.json(parsed)
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
