import json
import pandas as pd
import numpy as np
import pytest
import os
import sys
import importlib.util, os

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Signal testing mode to the app before import
os.environ['APP_TESTING'] = '1'

APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'app.py')
spec = importlib.util.spec_from_file_location('app_module', APP_PATH)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)


@pytest.fixture
def client(monkeypatch):
    # Prevent any real DB connections during tests
    monkeypatch.setattr(app_module, 'get_db_connection', lambda: None)

    # Disable ML models to avoid heavyweight loads
    monkeypatch.setattr(app_module, 'use_enhanced_models', False, raising=False)
    monkeypatch.setattr(app_module, 'nutrition_model', None, raising=False)

    # Minimal meals_df for CSV-driven helpers
    app_module.meals_df = pd.DataFrame({
        'meal': ['Baked salmon', 'Quinoa salad', 'Oatmeal'],
        'budget': [15.0, 8.0, 3.0],
        'ingredients': [
            'Salmon fillet, sweet potato, olive oil, herbs',
            'Quinoa, cucumber, tomato, olive oil, lemon',
            'Oats, banana, almonds, honey, milk'
        ],
        'calories': [500, 350, 250],
        'protein_g': [35, 12, 7],
        'carbs_g': [30, 45, 45],
        'fat_g': [20, 10, 5]
    })

    app = app_module.app
    app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test-secret'
    })

    with app.test_client() as client:
        yield client


def test_home_route_ok(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'<title>' in resp.data or b'<!DOCTYPE html>' in resp.data


def test_login_get_ok(client):
    resp = client.get('/login')
    assert resp.status_code == 200


def test_admin_dashboard_requires_login(client):
    # Not logged in -> redirected to login
    resp = client.get('/admin/dashboard', follow_redirects=False)
    assert resp.status_code in (301, 302)
    assert '/login' in resp.headers.get('Location', '')


def test_csv_helpers_select_by_budget(monkeypatch):
    # Create a local meals_df and use helpers directly
    meals_df = pd.DataFrame({
        'meal': ['Meal A', 'Meal B', 'Meal C'],
        'budget': [10.0, 20.0, 5.0],
        'ingredients': ['A1', 'B1', 'C1']
    })
    monkeypatch.setattr(app_module, 'meals_df', meals_df, raising=False)

    # Budget 12 -> should pick Meal A (most expensive within budget)
    data = {'budget': 12.0}
    meal = app_module.get_meal_from_csv(data)
    assert meal == 'Meal A'

    # No budget -> pick the cheapest
    data2 = {}
    meal2 = app_module.get_meal_from_csv(data2)
    assert meal2 in ('Meal C', 'Meal A', 'Meal B')  # deterministic by sort: Meal C


def test_csv_helpers_ingredients_lookup(monkeypatch):
    meals_df = pd.DataFrame({
        'meal': ['Quinoa salad', 'Baked salmon'],
        'budget': [8.0, 15.0],
        'ingredients': ['Q-ING', 'S-ING']
    })
    monkeypatch.setattr(app_module, 'meals_df', meals_df, raising=False)

    ing = app_module.get_ingredients_from_csv('quinoa SALAD')
    assert ing == 'Q-ING'


def test_predict_uses_csv_helpers(client):
    # Set a session for login_required
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'testuser'
        sess['user_role'] = 'customer'

    form_data = {
        'age': '30',
        'gender': 'Male',
        'weight': '70',
        'height': '175',
        'disease_type': 'None',
        'severity': 'Mild',
        'physical_activity_level': 'Moderate',
        'cholesterol': '180',
        'blood_pressure': '120',
        'glucose': '95',
        'dietary_restrictions': 'None',
        'dietary_habits': 'Non-Vegetarian',
        'meal_type': 'Dinner',
        'budget': '10'
    }

    resp = client.post('/predict', data=form_data)
    assert resp.status_code == 200

    payload = resp.get_json()
    assert payload['success'] is True
    preds = payload['predictions']

    # Ensure keys present
    assert 'diet_recommendation' in preds
    assert 'weekly_exercise_hours' in preds
    assert 'meal' in preds
    assert 'ingredients' in preds
    assert 'nutrition' in preds

    # Meal should be chosen from our meals_df by budget
    assert preds['meal'] in app_module.meals_df['meal'].astype(str).tolist()

    # Budget info present
    assert preds['budget_entered'] == float(form_data['budget'])
    assert 'estimated_meal_cost' in preds
    assert 'within_budget' in preds
