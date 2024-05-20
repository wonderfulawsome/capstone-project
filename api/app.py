from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    data = request.get_json()
    # 최적화 로직 구현
    result = {'message': f"Portfolio optimized based on input: {data}"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
