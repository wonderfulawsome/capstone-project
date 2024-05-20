from flask import Flask, request, jsonify
from flask_cors import CORS
from input import get_user_input
from data_preprocessed import preprocess_data
from data_download import download_data
from Clustering import execute_clustering

app = Flask(__name__)
CORS(app)  # CORS 설정

@app.route('/api/optimize', methods=['POST'])
def optimize():
    # 클라이언트로부터 JSON 데이터를 받습니다.
    data = request.get_json()

    # 입력 데이터를 전처리합니다.
    processed_data = preprocess_data(data)

    # 클러스터링을 수행합니다.
    result = execute_clustering(processed_data)

    # 결과를 JSON 형태로 반환합니다.
    return jsonify({"clustering_result": result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
