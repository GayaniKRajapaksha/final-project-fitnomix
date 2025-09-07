# tests/test_routes.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_register_page(client):
    response = client.get('/register')
    assert response.status_code == 200
    assert b"Register" in response.data or b"register" in response.data

def test_login_page(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b"Login" in response.data or b"login" in response.data
