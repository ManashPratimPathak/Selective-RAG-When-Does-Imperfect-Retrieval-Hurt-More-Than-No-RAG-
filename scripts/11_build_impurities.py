"""Build and freeze structured adversarial distractor passages for test questions.

Generates:
1. Entity-Overlap Distractors: Passages that share words/names with the question subject
   without answering it.
2. Relation-Matched Distractors: Passages describing other entities of the same relation
   with competing answers of the same semantic type.
3. Temporal / Conflicting Distractors: Passages presenting historical or superseded
   claims with explicit temporal markers.

Outputs:
- data/impurities/adversarial_passages.jsonl
- data/impurities/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

try:
    from .impurities import (
        build_relation_passage,
        build_temporal_conflict_passage,
        contains_any_answer,
        extract_entity_tokens,
    )
except ImportError:
    from impurities import (
        build_relation_passage,
        build_temporal_conflict_passage,
        contains_any_answer,
        extract_entity_tokens,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_entity_overlap_chunks(
    subject: str,
    gold_title: str,
    accepted_answers: list[str],
    chunks: list[dict[str, Any]],
    k: int = 2,
) -> list[dict[str, Any]]:
    """Find top-k chunks from corpus that maximize token overlap with subject."""
    tokens = set(extract_entity_tokens(subject))
    gold_title_lower = gold_title.lower()
    scored_chunks: list[tuple[int, dict[str, Any]]] = []

    for c in chunks:
        if c.get("page_title", "").lower() == gold_title_lower:
            continue
        text = c.get("text", "")
        if contains_any_answer(text, accepted_answers):
            continue
        chunk_tokens = set(extract_entity_tokens(c.get("page_title", "") + " " + text[:250]))
        overlap = len(tokens & chunk_tokens)
        if overlap > 0:
            scored_chunks.append((overlap, c))

    scored_chunks.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
    return [c for _, c in scored_chunks[:k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    sheet_path = root / "data/manifests/passage_verification_sheet.jsonl"
    chunks_path = root / "data/corpus/chunks.jsonl"
    output_path = root / "data/impurities/adversarial_passages.jsonl"
    manifest_path = root / "data/impurities/manifest.json"

    questions = [
        row for row in load_jsonl(sample_path)
        if row.get("split") == "test"
    ]
    reviews = {str(row["question_id"]): row for row in load_jsonl(sheet_path)}
    chunks = load_jsonl(chunks_path)

    # Load PopQA arrow database
    cached_arrows = sorted(Path.home().glob(".cache/huggingface/datasets/**/pop_qa-test.arrow"))
    if not cached_arrows:
        raise FileNotFoundError("pop_qa-test.arrow not found in cache")
    popqa_dataset = Dataset.from_file(str(cached_arrows[-1]))

    # Group PopQA rows by relation for relation and temporal distractors
    popqa_by_relation: dict[str, list[dict[str, Any]]] = {}
    for row in popqa_dataset:
        popqa_by_relation.setdefault(row["prop"], []).append(row)

    adversarial_records: list[dict[str, Any]] = []

    for idx, q in enumerate(questions):
        qid_str = str(q["question_id"])
        review = reviews[qid_str]
        subject = q["subject"]
        relation = q["relation"]
        accepted_answers = [str(a) for a in q.get("accepted_answers", [])]
        gold_title = q.get("wikipedia_title", "")

        # 1. Entity-Overlap Distractors
        overlap_chunks = find_entity_overlap_chunks(
            subject=subject,
            gold_title=gold_title,
            accepted_answers=accepted_answers,
            chunks=chunks,
            k=2,
        )
        if len(overlap_chunks) >= 2:
            e_doc1 = f"[{overlap_chunks[0]['chunk_id']}] {overlap_chunks[0]['page_title']}: {overlap_chunks[0]['text']}"
            e_doc2 = f"[{overlap_chunks[1]['chunk_id']}] {overlap_chunks[1]['page_title']}: {overlap_chunks[1]['text']}"
        elif len(overlap_chunks) == 1:
            e_doc1 = f"[{overlap_chunks[0]['chunk_id']}] {overlap_chunks[0]['page_title']}: {overlap_chunks[0]['text']}"
            # Construct a second entity-overlap passage using PopQA entity sharing tokens
            e_doc2 = f"[synthetic_overlap_{qid_str}_2] Context regarding {subject}: Documentation on related applications and terminology concerning {subject} across diverse fields."
        else:
            e_doc1 = f"[synthetic_overlap_{qid_str}_1] Context regarding {subject}: Documentation on related applications and terminology concerning {subject} across diverse fields."
            e_doc2 = f"[synthetic_overlap_{qid_str}_2] Analysis and historical definitions mentioning {subject} within technical reference glossaries."

        # 2. Relation-Matched Distractors
        # Sample candidates from PopQA with the exact same relation
        candidates = popqa_by_relation.get(relation, [])
        valid_candidates = [
            c for c in candidates
            if int(c["id"]) != q["question_id"]
            and not contains_any_answer(str(c["obj"]), accepted_answers)
            and c["subj"] != subject
        ]

        if len(valid_candidates) >= 2:
            c1 = valid_candidates[(idx * 2) % len(valid_candidates)]
            c2 = valid_candidates[(idx * 2 + 1) % len(valid_candidates)]
            r_doc1 = f"[rel_matched_{c1['id']}] {build_relation_passage(c1['subj'], relation, c1['obj'])}"
            r_doc2 = f"[rel_matched_{c2['id']}] {build_relation_passage(c2['subj'], relation, c2['obj'])}"
            competing_ans1 = str(c1["obj"])
            competing_ans2 = str(c2["obj"])
        else:
            r_doc1 = f"[rel_matched_gen_1] {build_relation_passage('Related work', relation, 'Alternative Entity')}"
            r_doc2 = f"[rel_matched_gen_2] {build_relation_passage('Secondary reference', relation, 'Secondary Entity')}"
            competing_ans1 = "Alternative Entity"
            competing_ans2 = "Secondary Entity"

        # 3. Temporal / Conflicting Distractor
        t_doc1 = f"[temporal_conflict_{qid_str}_1] {build_temporal_conflict_passage(subject, relation, competing_ans1, seed_offset=1)}"
        t_doc2 = f"[temporal_conflict_{qid_str}_2] {build_temporal_conflict_passage(subject, relation, competing_ans2, seed_offset=2)}"

        adversarial_records.append({
            "question_id": q["question_id"],
            "subject": subject,
            "relation": relation,
            "accepted_answers": accepted_answers,
            "clean_gold_doc": review["clean_doc_1"],
            "clean_gold_doc_2": review["clean_doc_2"],
            "entity_overlap_doc_1": e_doc1,
            "entity_overlap_doc_2": e_doc2,
            "relation_matched_doc_1": r_doc1,
            "relation_matched_doc_2": r_doc2,
            "temporal_doc_1": t_doc1,
            "temporal_doc_2": t_doc2,
        })

    write_jsonl(output_path, adversarial_records)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    manifest = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_test_questions": len(questions),
        "total_records": len(adversarial_records),
        "sha256": digest,
        "artifact": str(output_path.relative_to(root)).replace("\\", "/"),
        "impurity_types": [
            "entity_overlap",
            "relation_matched",
            "temporal_conflict",
            "combined_adversarial",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Successfully generated and froze {len(adversarial_records)} adversarial impurity records.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
