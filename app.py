from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    app.logger.info('Received data: %s', data)
    print(f"Received data: {data}")

    processed_data = preprocess_data(data)
    print(f"Processed data: {processed_data}")

    result = execute_clustering(processed_data)
    print(f"Clustering result: {result}")

    response = jsonify({"clustering_result": result.tolist()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

