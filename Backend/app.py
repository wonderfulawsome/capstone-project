from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()  # 요청에서 JSON 데이터를 가져옴
    selected_option = data['option']  # JSON 데이터에서 'option' 값을 추출
    # 실제 포트폴리오 최적화 로직을 여기에 구현
    # 예시로 선택된 옵션에 따라 다른 메시지를 반환
    response = {
        'message': f'Portfolio optimized based on option {selected_option}'
    }
    return jsonify(response)  # JSON 형태로 응답 반환

if __name__ == '__main__':
    app.run(debug=True)  # 서버를 디버그 모드로 실행
