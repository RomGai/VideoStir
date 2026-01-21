"""Evaluate LLaVA video sample logs for answer correctness."""

import argparse
import json
import os
from typing import Dict, List, Tuple

DEFAULT_LOG_PATH = os.getenv("LLAVA_RESULT_LOG", os.path.join("output", "llava_video_samples.json"))


def _normalize_choice(raw_choice: str) -> str:
    """Extract the first alphabetical character as the canonical option letter."""

    for ch in raw_choice.strip():
        if ch.isalpha():
            return ch.upper()
    return raw_choice.strip().upper()


def _load_entries(log_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, list):
        raise ValueError("Expected the log file to contain a JSON array of entries.")

    return data


def _evaluate_entries(entries: List[Dict[str, str]]) -> Tuple[int, int, List[Dict[str, str]]]:
    """Return (num_correct, total, mismatches) based on normalized comparisons."""

    num_correct = 0
    mismatches: List[Dict[str, str]] = []

    for entry in entries:
        correct_choice = str(entry.get("correct_choice", "")).strip()
        model_reply = str(entry.get("model_reply", "")).strip()

        normalized_correct = _normalize_choice(correct_choice) if correct_choice else ""
        normalized_reply = _normalize_choice(model_reply) if model_reply else ""
        is_match = bool(normalized_correct) and normalized_correct == normalized_reply

        if is_match:
            num_correct += 1
        else:
            mismatches.append(
                {
                    "sample_index": entry.get("sample_index"),
                    "id": entry.get("id"),
                    "correct_choice": correct_choice or "@",
                    "model_reply": model_reply or "@",
                    "normalized_correct_choice": normalized_correct or "@",
                    "normalized_model_reply": normalized_reply or "@",
                }
            )

    return num_correct, len(entries), mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH, help="Path to the JSON log produced by run_llava_video_samples.py"
    )
    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="Print details for samples where the model reply did not match the correct choice.",
    )
    args = parser.parse_args()

    entries = _load_entries(args.log)
    num_correct, total, mismatches = _evaluate_entries(entries)

    print(f"Total evaluated: {total}")
    print(f"Correct matches: {num_correct}")
    accuracy = (num_correct / total) * 100 if total else 0.0
    print(f"Accuracy: {accuracy:.2f}%")

    if args.show_mismatches and mismatches:
        print("\nMismatches:")
        for miss in mismatches:
            print(
                f"- Sample {miss.get('sample_index', '@')} (ID: {miss.get('id', '@')}):"
                f" expected {miss['normalized_correct_choice']}, got {miss['normalized_model_reply']}"
            )


if __name__ == "__main__":
    main()
