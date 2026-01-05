from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import threading

# Import existing RAG logic
# We need to make sure rag_query.py function 'query_rag' can return data instead of just printing
import rag_query

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Call the refactored function
        result = rag_query.query_rag_api(message)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing request: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure RAG index exists
    if not os.path.exists(rag_query.INDEX_TEXT_PATH) or not os.path.exists(rag_query.DB_PATH):
        print("Error: RAG Index not found. Please run 'python rag_ingest.py' first.")
    
    app.run(debug=True, port=5000)
