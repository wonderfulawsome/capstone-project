from flask import Flask, request, jsonify
from input import get_user_input
from data_preprocessed import preprocess_data
from data_download import download_data
from Clustering import execute_clustering

app = Flask(__name__)

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    user_input = get_user_input()  # 사용자 입력 받기 (예시)
    processed_data = preprocess_data(user_input)
    result = execute_clustering(processed_data)
    return jsonify({"clustering_result": result.tolist()})  # 리스트 형식으로 변환하여 반환

if __name__ == '__main__':
    app.run(debug=True)
