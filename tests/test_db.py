# tests/test_db.py
import pytest
import app  # import your app module

def test_db_connection_mock(monkeypatch):
    class MockConn:
        def is_connected(self): return True
        def cursor(self, dictionary=True):
            class Cursor:
                def execute(self, query): pass
                def fetchone(self): return {'DATABASE()': 'meal_planner'}
                def close(self): pass
            return Cursor()
        def close(self): pass

    monkeypatch.setattr("app.mysql.connector.connect", lambda **kwargs: MockConn())
    assert app.test_db_connection() is True
