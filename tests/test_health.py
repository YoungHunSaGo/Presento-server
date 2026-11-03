def test_health_ok(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_cors_allows_frontend_origin(client):
    res = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    res = client.get("/api/v1/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in res.headers


def test_openapi_available(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    assert "/api/v1/health" in res.json()["paths"]
