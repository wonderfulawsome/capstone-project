from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 추가
from input import get_user_input
from data_preprocessed import preprocess_data
from data_download import download_data
from Clustering import execute_clustering

app = Flask(__name__)
CORS(app)  # CORS 설정

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    user_input = get_user_input()
    processed_data = preprocess_data(user_input)
    result = execute_clustering(processed_data)
    return jsonify({"clustering_result": result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
