import json
from pathlib import Path

from scripts.build_bm25 import build_index


def test_build_index_saves_tokens_order_settings_parameters_and_manifest_hash(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"chunk_id": "c0", "text": "First, chunk!"},
                {"chunk_id": "c1", "text": "Second chunk 42."},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tokenized = tmp_path / "tokenized.jsonl"
    manifest = tmp_path / "manifest.json"

    result = build_index(chunks, tokenized, manifest, k1=1.5, b=0.75)

    rows = [json.loads(line) for line in tokenized.read_text(encoding="utf-8").splitlines()]
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["chunk_id"] for row in rows] == ["c0", "c1"]
    assert rows[0]["tokens"] == ["first", "chunk"]
    assert rows[1]["tokens"] == ["second", "chunk", "42"]
    assert saved["chunk_ids"] == ["c0", "c1"]
    assert saved["corpus_order"] == [0, 1]
    assert saved["tokenization"]["stemming"] is False
    assert saved["bm25_parameters"] == {"k1": 1.5, "b": 0.75}
    assert len(saved["corpus_manifest_sha256"]) == 64
    assert result.corpus_size == 2


def test_frozen_corpus_artifact_is_complete_and_rebuildable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data/corpus/chunks.jsonl"
    tokenized_path = root / "data/index/bm25_tokenized.jsonl"
    manifest_path = root / "data/manifests/bm25_manifest.json"
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenized = [json.loads(line) for line in tokenized_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [str(chunk["chunk_id"]) for chunk in chunks]

    assert len(chunks) == 21_858
    assert len(tokenized) == len(chunks) == manifest["corpus_size"]
    assert [row["chunk_id"] for row in tokenized] == expected_ids
    assert manifest["chunk_ids"] == expected_ids
    assert manifest["corpus_order"] == list(range(len(chunks)))
    assert len(set(expected_ids)) == len(expected_ids)
    assert len(manifest["chunk_ids"]) == len(set(manifest["chunk_ids"]))
    assert len(manifest["corpus_manifest_sha256"]) == 64
    assert len(manifest["corpus_sha256"]) == 64
    assert len(manifest["tokenized_corpus_sha256"]) == 64

    rebuilt_tokenized = tmp_path / "bm25_tokenized.jsonl"
    rebuilt_manifest = tmp_path / "bm25_manifest.json"
    from scripts.build_bm25 import build_index

    build_index(chunks_path, rebuilt_tokenized, rebuilt_manifest, k1=1.5, b=0.75)
    rebuilt = json.loads(rebuilt_manifest.read_text(encoding="utf-8"))
    assert rebuilt["chunk_ids"] == manifest["chunk_ids"]
    assert rebuilt["corpus_order"] == manifest["corpus_order"]
    assert rebuilt["corpus_manifest_sha256"] == manifest["corpus_manifest_sha256"]
    assert rebuilt["corpus_sha256"] == manifest["corpus_sha256"]
    assert rebuilt["tokenized_corpus_sha256"] == manifest["tokenized_corpus_sha256"]
