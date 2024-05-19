import unittest
from app import app, db, Data

class TestCase(unittest.TestCase):
    def setUp(self):
        # 애플리케이션 컨텍스트 푸시
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        # 데이터베이스 초기화
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        db.create_all()

    def tearDown(self):
        # 데이터베이스 세션 제거 및 드롭
        db.session.remove()
        db.drop_all()

        # 애플리케이션 컨텍스트 팝
        self.app_context.pop()

    def test_submit_data(self):
        # 데이터 제출 테스트
        response = self.app.post('/submit_data', json={'content': 'Test data'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Data Test data added successfully', response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()

