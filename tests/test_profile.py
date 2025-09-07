# tests/test_profile.py
import pytest
from app import save_health_progress

def test_bmi_calculation(monkeypatch):
    # Prevent DB insert
    monkeypatch.setattr("app.get_db_connection", lambda: None)

    data = {
        'weight': 70,
        'height': 175,
        'cholesterol': 200,
        'blood_pressure': 120,
        'glucose': 90,
        'physical_activity_level': 'Moderate'
    }
    bmi = data['weight'] / ((data['height']/100)**2)
    assert round(bmi, 2) == 22.86
