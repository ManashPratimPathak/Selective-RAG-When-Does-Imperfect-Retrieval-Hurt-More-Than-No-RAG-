import json
from pathlib import Path
from bs4 import BeautifulSoup

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data/manifests/distractor_manifest.json").read_text(encoding="utf-8"))
pages = {
    str(json.loads(line)["page_id"]): json.loads(line)
    for line in (root / "data/corpus/distractor_pages.jsonl").read_text(encoding="utf-8").splitlines()
}
for group in ("head", "tail"):
    reviewed = 0
    for item in manifest["pages"]:
        if item.get("status") != "fetched" or item.get("entity_group") != group or reviewed >= 10:
            continue
        page = pages.get(str(item["final_page_id"]))
        if page is None:
            continue
        soup = BeautifulSoup(page["raw_html"], "html.parser")
        heading = soup.find("h1")
        text = soup.get_text(" ", strip=True)
        print(json.dumps({
            "entity_group": group,
            "page_id": str(page["page_id"]),
            "page_title": page["page_title"],
            "h1_present": bool(heading),
            "html_nonempty": bool(page["raw_html"].strip()),
            "text_chars": len(text),
            "manual_result": "pass" if page["raw_html"].strip() and len(text) > 1000 else "review",
        }, ensure_ascii=False))
        reviewed += 1
