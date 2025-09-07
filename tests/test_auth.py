# tests/test_auth.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_register_post(monkeypatch, client):
    # Mock DB connection to avoid real insert
    class MockCursor:
        def execute(self, query, values): pass
        def close(self): pass
    class MockConn:
        def cursor(self, dictionary=True): return MockCursor()
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr("app.get_db_connection", lambda: MockConn())

    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'test@test.com',
        'password': '12345'
    }, follow_redirects=True)

    assert response.status_code == 200

def test_login_invalid_user(monkeypatch, client):
    # Mock DB returning no user
    class MockCursor:
        def execute(self, query, values): pass
        def fetchone(self): return None
        def close(self): pass
    class MockConn:
        def cursor(self, dictionary=True): return MockCursor()
        def close(self): pass

    monkeypatch.setattr("app.get_db_connection", lambda: MockConn())
    response = client.post('/login', data={'username':'nouser','password':'123'}, follow_redirects=True)
    assert b"Invalid username or password" in response.data
