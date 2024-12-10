import time
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from process_image import find_queens
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save the uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, 'file.jpg')
        file.save(input_path)

        # Generate a unique filename for the output
        unique_filename = f"processed.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, unique_filename)

        try:
            find_queens(input_path, output_path)

            # Add a timestamp to the URL to force a fresh fetch
            timestamp = int(time.time())
            return jsonify({
                "success": True,
                "image_url": f"/outputs/{unique_filename}?t={timestamp}"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
