import json, os, uuid, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

ALL_CANDIDATES = []
PREDICTIONS = {}
CAND_MAP = {}
CLASS_SUMMARY = {"total": 0, "nn_counts": {}}
analyze_light_curve = None

try:
    from predictions import predict_candidates, analyze_light_curve as _alc, classification_summary
    analyze_light_curve = _alc
    ALL_CANDIDATES, PREDICTIONS = predict_candidates()
    CAND_MAP = {}
    for c in ALL_CANDIDATES:
        tid = c.get("tic_id")
        if tid is not None:
            CAND_MAP[tid] = c
    CLASS_SUMMARY = classification_summary(candidates=ALL_CANDIDATES, predictions=PREDICTIONS)
except Exception as e:
    print(f" ! Model loading: {e}")


def load_json(rel_path):
    path = os.path.join(OUTPUTS, rel_path)
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(raw)


@app.route("/")
def index():
    return render_template(
        "index.html",
        candidates_json=json.dumps(ALL_CANDIDATES),
        summary=CLASS_SUMMARY,
    )


@app.route("/candidates")
def candidates_list():
    return render_template("candidates.html", candidates=ALL_CANDIDATES)


@app.route("/candidates/<int:tic_id>")
def candidate_detail(tic_id):
    c = CAND_MAP.get(tic_id)
    if c is None:
        return "<h1>Not found</h1><p><a href='/candidates'>Back</a></p>", 404
    return render_template("candidate_detail.html", candidate=c)


@app.route("/api/candidates")
def api_candidates():
    return jsonify(ALL_CANDIDATES)


@app.route("/api/predictions")
def api_predictions():
    return jsonify({str(k): v for k, v in PREDICTIONS.items()})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if analyze_light_curve is None:
        return jsonify({"error": "Models not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("parquet", "fits", "csv"):
        return jsonify({"error": "Use .parquet, .fits, or .csv"}), 400
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fid = uuid.uuid4().hex[:8]
    tmp_path = os.path.join(UPLOAD_DIR, f"upload_{fid}.{ext}")
    try:
        file.save(tmp_path)
        result = analyze_light_curve(tmp_path, ext)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" * {len(ALL_CANDIDATES)} candidates, {len(PREDICTIONS)} with predictions")
    print(f" * Summary: {CLASS_SUMMARY}")
    print(f" * Running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
