# tests/test_prediction.py
import pytest
import pandas as pd
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = "testuser"
            sess['user_role'] = "customer"
        yield client

def test_predict_route(monkeypatch, client):
    class MockPreprocessor:
        def transform(self, df): return df
    class MockRegressor:
        def predict(self, X): return [5.0]
    class MockClassifier:
        def predict(self, X): return [0]

    monkeypatch.setattr("app.preprocessor_X", MockPreprocessor())
    monkeypatch.setattr("app.regressor", MockRegressor())
    monkeypatch.setattr("app.classifiers", {
        'diet_recommendation': MockClassifier(),
        'meal': MockClassifier(),
        'ingredients': MockClassifier()
    })
    monkeypatch.setattr("app.label_encoders", {
        'diet_recommendation': type("LE", (), {"inverse_transform": lambda self, x: ["Balanced Diet"]})(),
        'meal': type("LE", (), {"inverse_transform": lambda self, x: ["Rice & Curry"]})(),
        'ingredients': type("LE", (), {"inverse_transform": lambda self, x: ["Rice, Vegetables"]})(),
    })
    monkeypatch.setattr("app.target_encoders", {
        'diet_recommendation': type("TE", (), {"categories_": [range(1)]})(),
        'meal': type("TE", (), {"categories_": [range(1)]})(),
        'ingredients': type("TE", (), {"categories_": [range(1)]})(),
    })
    monkeypatch.setattr("app.meals_df", pd.DataFrame({'meal': ['Rice & Curry'], 'budget':[500]}))

    response = client.post('/predict', data={
        'age': '25','gender':'M','weight':'70','height':'175',
        'disease_type':'None','severity':'Low','physical_activity_level':'Moderate',
        'cholesterol':'200','blood_pressure':'120','glucose':'90',
        'dietary_restrictions':'None','dietary_habits':'Normal',
        'meal_type':'Lunch','budget':'1000'
    })

    assert response.status_code == 200
    assert b"success" in response.data
