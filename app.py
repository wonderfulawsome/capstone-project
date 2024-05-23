from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask 애플리케이션 생성
app = Flask(__name__)

# CORS 설정 - 모든 도메인에서 접근을 허용하도록 설정
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    # 클라이언트로부터 JSON 데이터를 받습니다.
    data = request.get_json()
    
    # 수신된 데이터를 로그로 출력합니다.
    app.logger.info('Received data: %s', data)
    print(f"Received data: {data}")  # 로그 데이터 출력

    # 받은 데이터를 사용하여 전처리합니다.
    processed_data = preprocess_data(data)
    print(f"Processed data: {processed_data}")  # 전처리된 데이터 로그 출력

    # 클러스터링을 수행합니다.
    result = execute_clustering(processed_data)
    print(f"Clustering result: {result}")  # 클러스터링 결과 로그 출력

    # 결과를 JSON 형태로 반환합니다.
    return jsonify({"clustering_result": result.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

