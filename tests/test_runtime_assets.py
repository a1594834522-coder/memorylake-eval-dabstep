import json

from dabstep_agent_pydantic.runtime_assets import load_runtime_assets


def test_load_runtime_assets_computes_stable_content_fingerprint(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    route_cards_path = assets_dir / "route_cards.json"
    route_cards_path.write_text(json.dumps({"cards": []}), encoding="utf-8")

    first = load_runtime_assets(assets_dir).asset_fingerprint
    second = load_runtime_assets(assets_dir).asset_fingerprint

    assert first is not None
    assert first.startswith("sha256:")
    assert second == first

    route_cards_path.write_text(json.dumps({"cards": [{"route_id": "x", "title": "X"}]}), encoding="utf-8")

    assert load_runtime_assets(assets_dir).asset_fingerprint != first


def test_load_runtime_assets_fingerprints_document_only_directory(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "manual.md").write_text("public manual text", encoding="utf-8")

    assets = load_runtime_assets(assets_dir)

    assert assets.route_cards == []
    assert assets.asset_fingerprint is not None
    assert assets.asset_fingerprint.startswith("sha256:")
