from __future__ import annotations

import json
import ssl
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol
from urllib import request
from urllib.error import HTTPError
from urllib.error import URLError

from dabstep_agent_pydantic.memory_models import MemorySearchHit


DEFAULT_MEMORYLAKE_BASE_URL = "https://app.memorylake.ai"


class Transport(Protocol):
    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict | None = None,
        timeout: int = 30,
    ) -> dict:
        ...

    def put_bytes(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> dict[str, str]:
        ...


class UrllibTransport:
    def __init__(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep

    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict | None = None,
        timeout: int = 30,
    ) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(url, data=data, headers=headers, method=method)
        delays = (2.0, 5.0)
        for attempt in range(len(delays) + 1):
            try:
                with request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code < 500 or attempt == len(delays):
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"MemoryLake API error {exc.code}: {body}") from exc
                self._sleep(delays[attempt])
            except (URLError, ssl.SSLError):
                if attempt == len(delays):
                    raise
                self._sleep(delays[attempt])
        raise RuntimeError("unreachable retry loop state")

    def put_bytes(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> dict[str, str]:
        req = request.Request(url, data=payload, headers=headers or {}, method="PUT")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return dict(response.headers.items())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MemoryLake upload error {exc.code}: {body}") from exc


@dataclass
class MemoryLakeClient:
    api_key: str
    base_url: str = DEFAULT_MEMORYLAKE_BASE_URL
    transport: Transport | None = None
    timeout: int = 30

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.transport is None:
            self.transport = UrllibTransport()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def search_memories(
        self,
        project_id: str,
        *,
        user_id: str,
        query: str,
        top_k: int,
        threshold: float,
        rerank: bool,
    ) -> list[MemorySearchHit]:
        url = f"{self.base_url}/openapi/memorylake/api/v2/projects/{project_id}/memories/search"
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "threshold": threshold,
            "rerank": rerank,
        }
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        return [MemorySearchHit(**item) for item in response.get("data", {}).get("results", [])]

    def search_documents(
        self,
        project_id: str,
        *,
        query: str,
        top_n: int,
    ) -> list[dict]:
        url = f"{self.base_url}/openapi/memorylake/api/v1/projects/{project_id}/documents/search"
        payload = {"query": query, "top_n": top_n}
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        return list(response.get("data", {}).get("results", []))

    def search_project(
        self,
        project_id: str,
        *,
        user_id: str,
        query: str,
        top_n: int,
        threshold: float,
    ) -> dict[str, list]:
        url = f"{self.base_url}/openapi/memorylake/api/v1/projects/{project_id}/search"
        payload = {"query": query, "user_id": user_id, "top_n": top_n, "threshold": threshold}
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        data = response.get("data", {})
        return {
            "documents": list(data.get("documents", [])),
            "memories": [MemorySearchHit(**item) for item in data.get("memories", [])],
        }

    def list_memories(
        self,
        project_id: str,
        *,
        user_id: str | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        assert self.transport is not None
        items: list[dict] = []
        page = 1
        while True:
            url = f"{self.base_url}/openapi/memorylake/api/v2/projects/{project_id}/memories?page={page}&size={page_size}"
            if user_id:
                url += f"&user_id={user_id}"
            response = self.transport.request_json("GET", url, self._headers(), None, self.timeout)
            data = response.get("data", {})
            batch = list(data.get("items", []))
            items.extend(batch)
            total_pages = int(data.get("total_pages", 1) or 1)
            if page >= total_pages or not batch:
                return items
            page += 1

    def add_memory(
        self,
        project_id: str,
        *,
        messages: list[dict[str, str]],
        user_id: str,
        chat_session_id: str,
        metadata: dict[str, str],
        infer: bool = True,
    ) -> list[str]:
        url = f"{self.base_url}/openapi/memorylake/api/v2/projects/{project_id}/memories"
        payload = {
            "messages": messages,
            "user_id": user_id,
            "chat_session_id": chat_session_id,
            "metadata": metadata,
            "infer": infer,
        }
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        return [
            str(item["event_id"])
            for item in response.get("data", {}).get("results", [])
            if item.get("event_id")
        ]

    def create_project(self, name: str, *, description: str = "") -> str:
        """Create a MemoryLake project and return its id (proj-...)."""
        url = f"{self.base_url}/openapi/memorylake/api/v1/projects"
        assert self.transport is not None
        response = self.transport.request_json(
            "POST", url, self._headers(), {"name": name, "description": description}, self.timeout
        )
        project_id = str(response.get("data", {}).get("id", ""))
        if not project_id:
            raise RuntimeError(f"MemoryLake project creation returned no id: {response}")
        return project_id

    def create_upload(self, file_size: int) -> dict:
        url = f"{self.base_url}/openapi/memorylake/api/v1/drives/items/upload"
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), {"file_size": file_size}, self.timeout)
        return dict(response.get("data", {}))

    def create_file_item(
        self,
        *,
        name: str,
        parent_item_id: str,
        upload_id: str,
        part_etags: list[dict[str, object]],
        conflict_strategy: str = "rename",
    ) -> str:
        url = f"{self.base_url}/openapi/memorylake/api/v1/drives/items"
        payload = {
            "item_type": "file",
            "parent_item_id": parent_item_id,
            "name": name,
            "from": {"upload_id": upload_id, "part_etags": part_etags},
            "conflict_strategy": conflict_strategy,
        }
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        return str(response.get("data", {}).get("item_id", ""))

    def upload_file_to_library(
        self,
        path: str | Path,
        *,
        parent_item_id: str = "MY_SPACE",
        conflict_strategy: str = "rename",
    ) -> str:
        path = Path(path)
        upload = self.create_upload(path.stat().st_size)
        part_etags: list[dict[str, object]] = []
        assert self.transport is not None
        with path.open("rb") as handle:
            for part in upload.get("part_items", []):
                chunk = handle.read(int(part["size"]))
                headers = self.transport.put_bytes(str(part["upload_url"]), chunk, {}, self.timeout)
                etag = headers.get("ETag") or headers.get("etag")
                if not etag:
                    raise RuntimeError("MemoryLake upload chunk response did not include ETag")
                part_etags.append({"number": int(part["number"]), "etag": etag})
        return self.create_file_item(
            name=path.name,
            parent_item_id=parent_item_id,
            upload_id=str(upload["upload_id"]),
            part_etags=part_etags,
            conflict_strategy=conflict_strategy,
        )

    def add_documents(self, project_id: str, *, drive_item_ids: list[str]) -> dict[str, int]:
        url = f"{self.base_url}/openapi/memorylake/api/v1/projects/{project_id}/documents"
        payload = {"drive_item_ids": drive_item_ids}
        assert self.transport is not None
        response = self.transport.request_json("POST", url, self._headers(), payload, self.timeout)
        data = response.get("data", {})
        return {
            "success_count": int(data.get("success_count", 0)),
            "failure_count": int(data.get("failure_count", 0)),
        }
