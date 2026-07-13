import json
import ssl
from urllib.error import HTTPError
from urllib.error import URLError

import pytest

from dabstep_agent_pydantic.memorylake import UrllibTransport


class _JsonResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_urllib_transport_retries_transient_http_5xx(monkeypatch):
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        if len(calls) == 1:
            raise HTTPError(req.full_url, 503, "unavailable", hdrs={}, fp=None)
        return _JsonResponse({"success": True, "data": {"ok": True}})

    monkeypatch.setattr("dabstep_agent_pydantic.memorylake.request.urlopen", fake_urlopen)
    transport = UrllibTransport(sleep=sleeps.append)

    result = transport.request_json("POST", "https://memory.example/search", {"x": "y"}, {"q": "fee"})

    assert result == {"success": True, "data": {"ok": True}}
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_urllib_transport_retries_url_and_ssl_errors(monkeypatch):
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise URLError("temporary DNS failure")
        if len(calls) == 2:
            raise ssl.SSLError("hostname mismatch")
        return _JsonResponse({"success": True})

    monkeypatch.setattr("dabstep_agent_pydantic.memorylake.request.urlopen", fake_urlopen)
    transport = UrllibTransport(sleep=sleeps.append)

    result = transport.request_json("GET", "https://memory.example/search", {})

    assert result == {"success": True}
    assert len(calls) == 3
    assert sleeps == [2.0, 5.0]


def test_urllib_transport_does_not_retry_4xx(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise HTTPError(req.full_url, 401, "unauthorized", hdrs={}, fp=None)

    monkeypatch.setattr("dabstep_agent_pydantic.memorylake.request.urlopen", fake_urlopen)
    transport = UrllibTransport(sleep=lambda delay: None)

    with pytest.raises(RuntimeError, match="MemoryLake API error 401"):
        transport.request_json("GET", "https://memory.example/search", {})

    assert len(calls) == 1


def test_create_project_returns_id():
    class _Transport:
        def request_json(self, method, url, headers, payload=None, timeout=30):
            assert method == "POST" and url.endswith("/api/v1/projects")
            assert payload == {"name": "n", "description": ""}
            return {"success": True, "data": {"id": "proj-abc123"}}

        def put_bytes(self, *a, **k):
            raise AssertionError

    from dabstep_agent_pydantic.memorylake import MemoryLakeClient
    client = MemoryLakeClient(api_key="k", transport=_Transport())
    assert client.create_project("n") == "proj-abc123"
