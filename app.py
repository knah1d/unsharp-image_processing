import base64
import io
import os
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from enhance import build_comparison_image, compute_metrics, enhance, srgan_is_trained

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

OUTPUTS = "outputs"
os.makedirs(UPLOADS := "uploads", exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def _build_histogram_data(img_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    return hist


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": "Invalid or unsupported file"}), 400

    # Read parameters from form
    gamma      = float(request.form.get("gamma",      0.8))
    clip_limit = float(request.form.get("clip_limit", 2.0))
    cn         = float(request.form.get("cn",         0.85))
    sigma      = float(request.form.get("sigma",      1.0))
    use_srgan  = request.form.get("use_srgan", "true").lower() == "true"

    # Decode image
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Resize to 512×512 (paper uses 512×512 reshaped images)
    img_resized = cv2.resize(img, (512, 512))

    # Run enhancement pipeline
    enhanced = enhance(
        img_resized,
        gamma=gamma,
        clip_limit=clip_limit,
        cn=cn,
        sigma=sigma,
        use_srgan=use_srgan,
    )

    # Metrics
    metrics = compute_metrics(img_resized, enhanced)

    # Build comparison image (images + histograms in one frame)
    comparison = build_comparison_image(img_resized, enhanced, metrics)

    # Save files
    uid = uuid.uuid4().hex
    cv2.imwrite(os.path.join(OUTPUTS, f"{uid}_enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(OUTPUTS, f"{uid}_comparison.png"), comparison)

    return jsonify(
        {
            "original":   _to_b64(img_resized),
            "enhanced":   _to_b64(enhanced),
            "comparison": _to_b64(comparison),
            "metrics":    metrics,
            "srgan_trained": srgan_is_trained() if use_srgan else None,
            "dl_enhanced":   f"{uid}_enhanced.png",
            "dl_comparison": f"{uid}_comparison.png",
        }
    )


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUTS, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="image/png",
                     download_name="enhanced.png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
