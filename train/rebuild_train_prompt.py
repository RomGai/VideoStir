import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_frame_relevance_prompt(query: str) -> str:
    query = (query or "").strip()
    return (
        "Given the image, which is a frame from a video, rate how relevant this frame is for "
        f"answering the question: '{query}'.\n"
        "Output only one number from 1 to 5, where:\n"
        "1 = completely irrelevant — the frame provides no visual or contextual information related to the "
        "question or its answer.\n"
        "2 = slightly relevant — the frame shows general background or context, but it is unlikely to "
        "contribute to answering.\n"
        "3 = moderately relevant — the frame includes partial clues or indirect context that might help "
        "infer the answer, but the key evidence is missing.\n"
        "4 = mostly relevant — the frame provides substantial visual or contextual information that can be "
        "used to answer the question, though not fully decisive.\n"
        "5 = highly relevant — the frame clearly contains the decisive evidence or strong contextual cues "
        "that directly or indirectly support the correct answer."
    )


def _extract_query(text: str) -> str:
    text = (text or "").strip()
    marker = "answering the question: '"
    if text.startswith("Given the image, which is a frame from a video") and marker in text:
        start = text.find(marker) + len(marker)
        end = text.find("'.", start)
        if end > start:
            return text[start:end]
    return text


def rebuild_train_prompts(records: List[List[Dict[str, Any]]]) -> int:
    updated = 0
    for conversation in records:
        for message in conversation:
            if message.get("role") != "user":
                continue
            for item in message.get("content", []):
                if item.get("type") != "text":
                    continue
                query = _extract_query(item.get("text", ""))
                new_prompt = build_frame_relevance_prompt(query)
                if item.get("text") != new_prompt:
                    item["text"] = new_prompt
                    updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite train.json user text into reranker prompt format."
    )
    parser.add_argument("--input", default="train.json", help="Input JSON path")
    parser.add_argument("--output", default="train.json", help="Output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    updated = rebuild_train_prompts(records)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(f"Updated {updated} user text entries. Saved to {output_path}.")


if __name__ == "__main__":
    main()
