import json

from dabstep_agent_pydantic.memory_export import content_hash
from dabstep_agent_pydantic.memory_export import export_memories
from dabstep_agent_pydantic.memorylake import MemoryLakeClient


class FakePagedTransport:
    """Serves two pages of memories through the real client pagination loop."""

    def __init__(self):
        self.pages = {
            1: {
                "items": [
                    {"id": "mem-b", "content": "rule b", "user_id": "u", "expired": False,
                     "created_at": "2026-07-01", "updated_at": "2026-07-01", "extra": "dropped"},
                ],
                "total_pages": 2,
            },
            2: {
                "items": [
                    {"id": "mem-a", "content": "rule a", "user_id": "u", "expired": False,
                     "created_at": "2026-07-02", "updated_at": "2026-07-02"},
                ],
                "total_pages": 2,
            },
        }
        self.urls = []

    def request_json(self, method, url, headers, payload=None, timeout=30):
        assert method == "GET"
        self.urls.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        return {"success": True, "data": self.pages[page]}

    def put_bytes(self, url, payload, headers=None, timeout=30):
        raise AssertionError("unused")


def test_export_memories_paginates_sorts_and_hashes(tmp_path):
    transport = FakePagedTransport()
    client = MemoryLakeClient(api_key="key", transport=transport)
    output_path = tmp_path / "memory_export.jsonl"

    report = export_memories(client, project_id="project", user_id="u", output_path=output_path)

    assert len(transport.urls) == 2
    assert all("user_id=u" in url for url in transport.urls)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["mem-a", "mem-b"]
    assert "extra" not in rows[1]
    assert report["memory_count"] == 2
    assert report["content_sha256"] == content_hash(rows)


def test_content_hash_depends_only_on_content_sequence():
    rows_a = [{"id": "1", "content": "alpha"}, {"id": "2", "content": "beta"}]
    rows_b = [{"id": "9", "content": "alpha", "created_at": "x"}, {"id": "8", "content": "beta"}]
    assert content_hash(rows_a) == content_hash(rows_b)
    assert content_hash(rows_a) != content_hash(list(reversed(rows_a)))
