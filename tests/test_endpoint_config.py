import asyncio
import os

import httpx

from dabstep_agent_pydantic.agent import _RetryingAsyncTransport
from dabstep_agent_pydantic.agent import build_model_from_env
from dabstep_agent_pydantic.agent import resolve_model_endpoint_from_env
from dabstep_agent_pydantic.cli import build_memory_client
from dabstep_agent_pydantic.cli import configure_model_endpoint_env
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import RunMode


MODEL_ENV_KEYS = [
    "DABSTEP_MEMORY_ROUTER_MODE",
    "DABSTEP_MODEL",
    "DABSTEP_OPENAI_API_KEY",
    "DABSTEP_OPENAI_BASE_URL",
    "DABSTEP_MEMORY_ROUTER_API_KEY",
    "DABSTEP_MEMORY_ROUTER_BASE_URL",
    "MEMORYLAKE_OPENAI_BASE_URL",
    "MEMORYLAKE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
]


def test_clean_mode_accepts_dabstep_scoped_openai_endpoint(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DABSTEP_OPENAI_API_KEY", "validator-key")
    monkeypatch.setenv("DABSTEP_OPENAI_BASE_URL", "https://validator.example/v1")
    monkeypatch.setenv("DABSTEP_MODEL", "validator-model")

    endpoint = resolve_model_endpoint_from_env()

    assert endpoint.model_name == "validator-model"
    assert endpoint.api_key == "validator-key"
    assert endpoint.base_url == "https://validator.example/v1"
    assert endpoint.headers is None


def test_memory_router_base_url_and_api_key_are_configurable(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_MODE", "hosted")
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_API_KEY", "router-key")
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_BASE_URL", "https://router.example/v1")

    endpoint = resolve_model_endpoint_from_env()

    assert endpoint.api_key == "router-key"
    assert endpoint.base_url == "https://router.example/v1"
    assert endpoint.headers is None


def test_byok_router_uses_custom_openai_endpoint_with_memorylake_header(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_MODE", "byok")
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_BASE_URL", "https://router.example/v1/openai")
    monkeypatch.setenv("DABSTEP_OPENAI_API_KEY", "validator-openai-key")
    monkeypatch.setenv("MEMORYLAKE_API_KEY", "memorylake-key")

    endpoint = resolve_model_endpoint_from_env()

    assert endpoint.api_key == "validator-openai-key"
    assert endpoint.base_url == "https://router.example/v1/openai"
    assert endpoint.headers == {"x-memorylake-api-key": "memorylake-key"}


def test_memorylake_api_base_url_is_configurable(monkeypatch):
    monkeypatch.setenv("MEMORYLAKE_API_KEY", "memorylake-key")
    monkeypatch.setenv("MEMORYLAKE_BASE_URL", "https://memory.example")
    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        project_id="project",
        user_id="user",
    )

    client = build_memory_client(config)

    assert client is not None
    assert client.base_url == "https://memory.example"


def test_cli_router_mode_updates_model_endpoint_env(monkeypatch):
    monkeypatch.delenv("DABSTEP_MEMORY_ROUTER_MODE", raising=False)

    configure_model_endpoint_env("hosted")

    assert os.environ["DABSTEP_MEMORY_ROUTER_MODE"] == "hosted"


def test_model_transport_retries_transient_502_before_success():
    attempts = []
    sleeps = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(await request.aread())
        if len(attempts) == 1:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json={"ok": True})

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run_request() -> httpx.Response:
        transport = _RetryingAsyncTransport(httpx.MockTransport(handler), sleep=fake_sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://gateway.example/v1/chat/completions", json={"model": "validator"})

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert sleeps == [5.0]


def test_model_transport_uses_default_backoff_schedule_for_transient_gateway_errors():
    attempts = 0
    sleeps = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json={"ok": True})

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run_request() -> httpx.Response:
        transport = _RetryingAsyncTransport(httpx.MockTransport(handler), sleep=fake_sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://gateway.example/v1/chat/completions", json={"model": "validator"})

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert attempts == 4
    assert sleeps == [5.0, 15.0, 45.0]


def test_model_transport_does_not_retry_4xx_responses():
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"error": "missing API key"})

    async def run_request() -> httpx.Response:
        transport = _RetryingAsyncTransport(httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://gateway.example/v1/chat/completions", json={"model": "validator"})

    response = asyncio.run(run_request())

    assert response.status_code == 401
    assert attempts == 1


def test_model_transport_retries_connection_errors_before_success():
    attempts = 0
    sleeps = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("gateway refused connection", request=request)
        return httpx.Response(200, json={"ok": True})

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run_request() -> httpx.Response:
        transport = _RetryingAsyncTransport(httpx.MockTransport(handler), sleep=fake_sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://gateway.example/v1/chat/completions", json={"model": "validator"})

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert attempts == 2
    assert sleeps == [5.0]


def test_build_model_from_env_uses_retry_transport_and_closes_client(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DABSTEP_OPENAI_API_KEY", "validator-key")
    monkeypatch.setenv("DABSTEP_OPENAI_BASE_URL", "https://validator.example/v1")
    monkeypatch.setenv("DABSTEP_MODEL", "validator-model")

    async def build_and_close() -> tuple[bool, bool]:
        model = build_model_from_env()
        http_client = model.provider.client._client
        has_retry_transport = isinstance(http_client._transport, _RetryingAsyncTransport)
        async with model:
            pass
        return has_retry_transport, http_client.is_closed

    has_retry_transport, client_closed = asyncio.run(build_and_close())

    assert has_retry_transport
    assert client_closed


def _clear_env(monkeypatch):
    for key in MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_teacher_model_overrides_fall_back_to_solver_endpoint(monkeypatch):
    import dabstep_agent_pydantic.agent as agent_module
    from dabstep_agent_pydantic.agent import build_teacher_model_from_env

    monkeypatch.setattr(agent_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DABSTEP_MEMORY_ROUTER_MODE", "off")
    monkeypatch.setenv("DABSTEP_OPENAI_API_KEY", "solver-key")
    monkeypatch.setenv("DABSTEP_OPENAI_BASE_URL", "http://solver.example/v1")
    monkeypatch.setenv("DABSTEP_MODEL", "solver-model")
    monkeypatch.delenv("DABSTEP_TEACHER_MODEL", raising=False)
    monkeypatch.delenv("DABSTEP_TEACHER_BASE_URL", raising=False)
    monkeypatch.delenv("DABSTEP_TEACHER_API_KEY", raising=False)
    assert build_teacher_model_from_env().model_name == "solver-model"

    monkeypatch.setenv("DABSTEP_TEACHER_MODEL", "teacher-model")
    model = build_teacher_model_from_env()
    assert model.model_name == "teacher-model"
