"""Build a manually reviewed test-set error-analysis sheet."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CASES = {
    "damage": {
        1630183: ("Retrieval failure", "No verified supporting passage appears in the top three; the retrieved context does not anchor the director.",),
        1488536: ("Retrieval failure", "Top-three passages omit verified composer evidence; the imperfect answer switches to an unrelated Hamilton name.",),
        156556: ("Retrieval failure", "Top-three passages contain no verified genre evidence, and the answer changes from rock to jazz.",),
        2281514: ("Conflicting context", "Top-three context mixes John the Apostle and Jesus passages; the answer expands a correct short alias into a non-exact phrase.",),
        1446352: ("Conflicting context", "A verified passage is present, but the answer chooses the narrower State Shinto wording rather than an accepted alias.",),
        5777300: ("Retrieval failure", "Top-three results include an unrelated musician page and no verified genre evidence; the correct R&B answer is replaced.",),
    },
    "rescue": {
        2125444: ("Retrieval failure", "No verified support is in the top three; the model nevertheless supplies the accepted heavy-metal answer.",),
        1600144: ("Evidence ignored", "The top-ranked passage explicitly names Kevin Bray; retrieval succeeds and the imperfect answer uses it.",),
        2010305: ("Evidence ignored", "A verified top-ranked passage supports the documentary classification; retrieval corrects the No-RAG miss.",),
        6398386: ("Retrieval failure", "No verified support is in the top three, but the answer recovers Rob Zombie from retrieved page context.",),
        3213134: ("Evidence ignored", "A verified retrieved passage is present, and the imperfect answer recovers Irwin Winkler after a No-RAG miss.",),
        6547647: ("Retrieval failure", "No verified support appears in the top three; retrieval still changes the answer to United States.",),
        4145108: ("Retrieval failure", "Top-three passages lack verified father evidence; the imperfect answer nevertheless recovers Billy Ray Cyrus.",),
        1423107: ("Retrieval failure", "No verified support appears in the top three, but retrieved context leads to Emma Donoghue.",),
        6415969: ("Retrieval failure", "Top-three results are mixed and lack verified support; the imperfect answer recovers Boaz Yakin.",),
        1163745: ("Retrieval failure", "No verified support appears in the top three; the imperfect answer supplies Thomas D'Alesandro Jr.",),
    },
    "stable_failure": {
        534526: ("Retrieval failure", "No verified director evidence appears in the top three; both answers are unsupported names.",),
        535313: ("Ambiguous question", "The retrieved page names three directors, while the gold set contains only David Bruckner; answer selection is under-specified.",),
        3466481: ("Retrieval failure", "Top-three passages lack verified composer evidence; the imperfect answer is a refusal rather than a fact.",),
        4831102: ("Evidence ignored", "A verified top-ranked passage is retrieved, but neither answer matches the accepted thriller aliases.",),
        2658183: ("Ambiguous question", "Cambridge is geographically ambiguous without a country qualifier; England is defensible but not the accepted United Kingdom alias.",),
        4994642: ("Evidence ignored", "A verified top-ranked page is retrieved, but the model answers Mystery rather than the accepted soap-opera alias.",),
        1297727: ("Parametric failure", "No-RAG returns the prompt fallback and the top three have no verified producer evidence.",),
        3220050: ("Distractor override", "Top-three retrieval mixes The Kingdom with Troy and a composer passage; the imperfect answer follows an unrelated director.",),
        503138: ("Answer-format error", "The imperfect answer contains the correct producer plus an extra co-producer, so exact match fails.",),
        3593682: ("Evidence ignored", "The top-ranked page is the correct band page, but neither answer matches the accepted K-pop aliases.",),
    },
    "scoring_disagreement": {
        54966: ("Answer-format error", "Imperfect answer includes the accepted occupation physician inside a longer phrase; exact equality rejects the extra wording.",),
        13600: ("Answer-format error", "Imperfect answer contains the accepted Hoysala Empire answer plus historical qualifiers; no score change made.",),
        276787: ("Answer-format error", "Imperfect answer lists talent manager among several roles; normalized exact match rejects the multi-role response.",),
        6414257: ("Alias/scoring error", "Melodic death metal semantically contains the accepted death-metal genre, but no alias was added and exact scoring is unchanged.",),
        3111536: ("Alias/scoring error", "Roman Catholic is semantically close to the accepted Catholic Church aliases, but no alias was added and exact scoring is unchanged.",),
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    questions = {row["question_id"]: row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "test"}
    ledger = read_jsonl(root / "outputs/test_scoring_ledger.jsonl")
    answers: dict[int, dict[str, dict[str, Any]]] = {}
    for row in ledger:
        answers.setdefault(row["question_id"], {})[row["condition"]] = row
    retrieval = {}
    for row in read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl"):
        if row["rank"] <= 3:
            retrieval.setdefault(row["question_id"], []).append(row)

    output_rows: list[dict[str, Any]] = []
    for transition, selected in CASES.items():
        for question_id, (category, notes) in selected.items():
            question = questions[question_id]
            docs = retrieval.get(question_id, [])
            output_rows.append({
                "question_id": question_id,
                "group": question["popularity_group"],
                "relation": question["relation"],
                "no_rag_answer": answers[question_id]["no_rag"]["raw_prediction"],
                "imperfect_answer": answers[question_id]["imperfect_rag"]["raw_prediction"],
                "gold_aliases": json.dumps(question["accepted_answers"], ensure_ascii=False),
                "retrieved_documents": json.dumps([
                    {
                        "rank": doc["rank"],
                        "chunk_id": doc["chunk_id"],
                        "page_title": doc["page_title"],
                        "verified_support": doc["is_verified_support"],
                        "excerpt": doc["chunk_text"][:180],
                    }
                    for doc in docs
                ], ensure_ascii=False),
                "retrieval_success": any(doc["is_verified_support"] for doc in docs),
                "transition_type": transition,
                "error_category": category,
                "notes": notes,
            })

    output = root / "outputs/error_analysis.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (root / "outputs/error_analysis_alias_audit.json").write_text(
        json.dumps({"alias_changes": [], "note": "No aliases were added or scores changed during error analysis."}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output_rows), "artifact": str(output), "alias_changes": 0}, indent=2))


if __name__ == "__main__":
    main()
