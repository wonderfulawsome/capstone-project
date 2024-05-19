from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# 데이터베이스 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
db = SQLAlchemy(app)

# 데이터 모델
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return '<Data %r>' % self.content

# 데이터 제출 API
@app.route('/submit_data', methods=['POST'])
def submit_data():
    content = request.json['content']
    data = Data(content=content)
    db.session.add(data)
    db.session.commit()
    return jsonify({'message': f'Data {content} added successfully'})

if __name__ == "__main__":
    db.create_all()  # 데이터베이스 테이블 생성
    app.run(debug=True)

