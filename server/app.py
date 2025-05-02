from flask import Flask, request, jsonify
import json
from utils import (
    extract_text_from_pdf_from_url,
    chunk_by_sections,
    create_embeddings,
    search_bills,
    extract_bill_id
)
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

chunks = []
embeddings = {}

@app.route('/check-embeddings', methods=['GET'])
def check_embeddings():
    exists = os.path.exists("embeddings.json") and os.path.exists("chunks.json")
    return jsonify({"exists": exists}), 200

@app.route('/test', methods=['GET'])
def test_cors():
    return jsonify({"message": "CORS is working!"})

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    global chunks, embeddings
    data = request.get_json()
    pdf_url = data.get("pdf_url")

    try:
        text = extract_text_from_pdf_from_url(pdf_url)
        chunks = chunk_by_sections(text)
        embeddings = create_embeddings(chunks)

        with open("embeddings.json", "w") as f:
            json.dump(embeddings, f)

        with open("chunks.json", "w") as f:
            json.dump(chunks, f)

        return jsonify({"message": "Embeddings and chunks generated and saved."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    global chunks, embeddings
    data = request.get_json()
    query = data.get("query")
    try:
        # Load from file if not in memory
        if not embeddings:
            with open("embeddings.json", "r") as f:
                embeddings = json.load(f)
        if not chunks:
            with open("chunks.json", "r") as f:
                chunks = json.load(f)

        matched_chunks = search_bills(query, embeddings, chunks)
        bill_ids = [extract_bill_id(text) for _, text in matched_chunks]
        bill_ids = [b for b in bill_ids if b]
        unique_bill_ids = list(set(bill_ids))

        return jsonify({"bill_ids": unique_bill_ids}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
