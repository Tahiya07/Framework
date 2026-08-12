from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_and_protected_denial_without_loading_models():
    assert client.get("/health").json()["status"] == "ok"
    login = client.post("/auth/session", json={"role": "student", "access_code": "student-local"})
    token = login.json()["access_token"]
    r = client.post("/documents/index", headers={"Authorization": f"Bearer {token}"}, json={"text":"secret", "name":"exam", "scope":"protected"})
    assert r.status_code == 403
