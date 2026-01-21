from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch

import json
from difflib import SequenceMatcher
from typing import Dict, List, Any, Tuple, Optional

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(
    frames,
    frame_timestamps,
    subtitles,
    starting_timestamp_for_subtitles,
    duration,
    deduplicate_adjacent: bool = True,
):
    def _append_item(target_list: List[Any], item: Any):
        if not deduplicate_adjacent:
            target_list.append(item)
            return

        if not target_list:
            target_list.append(item)
            return

        last_item = target_list[-1]
        if isinstance(last_item, str) and isinstance(item, str):
            if last_item.strip() == item.strip():
                return
        target_list.append(item)

    interleaved_list: List[Any] = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles


            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]


        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    _append_item(interleaved_list, frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            _append_item(interleaved_list, subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)

    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        _append_item(interleaved_list, frame)

    return interleaved_list
    
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gather_subtitles(selected_segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    subtitles: List[Dict[str, Any]] = []
    max_end = 0.0
    for segment in selected_segments:
        end = float(segment.get("end_sec") or 0.0)
        max_end = max(max_end, end)
        for entry in segment.get("subtitle_entries", []) or []:
            entry_start = float(entry.get("start") or 0.0)
            entry_end = float(entry.get("end") or entry_start)
            subtitles.append(
                {
                    "timestamp": (entry_start, entry_end),
                    "text": entry.get("text") or entry.get("line") or "",
                }
            )
    subtitles.sort(key=lambda x: x["timestamp"][0])
    return subtitles, max_end


def _parse_choice_index(value: Any) -> Optional[int]:
    """Normalize various ``correct_choice`` encodings to a zero-based index."""

    if value is None:
        return None

    # Common case: already an int or an int-like string.
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        pass
    else:
        return int_value

    # Allow letter forms ("A" / "b" → 0 / 1).
    if isinstance(value, str) and len(value.strip()) == 1 and value.strip().isalpha():
        return ord(value.strip().upper()) - ord("A")

    return None


def _load_batch_correct_choices(output_root: str) -> Dict[str, int]:
    """Load correct choice metadata from the batch summary when available."""

    summary_path = os.path.join(output_root, "batch_results.json")
    if not os.path.exists(summary_path):
        return {}

    try:
        summary = _load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(summary, dict):
        return {}

    choices: Dict[str, int] = {}
    for key, value in summary.items():
        if not isinstance(value, dict):
            continue

        choice = _parse_choice_index(value.get("correct_choice"))
        if choice is None:
            continue

        choices[str(key)] = choice

    return choices


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _subtitle_similarity(target: str, candidate: str) -> float:
    target_norm = _normalize_text(target)
    candidate_norm = _normalize_text(candidate)
    if not target_norm or not candidate_norm:
        return 0.0
    matcher = SequenceMatcher(None, target_norm, candidate_norm)
    return matcher.ratio()


def _find_best_subtitle_match(
    subtitles: List[Dict[str, Any]], subtitle_query: str
) -> Optional[Dict[str, Any]]:
    if not subtitle_query:
        return None
    best_subtitle: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for entry in subtitles:
        score = _subtitle_similarity(subtitle_query, entry.get("text") or "")
        if score > best_score:
            best_score = score
            best_subtitle = entry
    return best_subtitle


def _dedupe_and_normalize_frames(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    normalized: List[Dict[str, Any]] = []
    for item in paths:
        frame_path = item.get("output_path") or item.get("path")
        if not frame_path:
            continue
        ts = float(item.get("timestamp") or 0.0)
        key = (frame_path, ts)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"output_path": frame_path, "timestamp": ts, **item})
    return normalized


def _sample_frames_in_range(
    frames: List[Dict[str, Any]], start: float, end: float, count: int
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    candidates = [f for f in frames if start <= float(f.get("timestamp", 0.0)) <= end]
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[i] for i in indices]


def _pick_frame_at_timestamp(
    frames: List[Dict[str, Any]], target_ts: float
) -> Optional[Dict[str, Any]]:
    later_frames = [f for f in frames if float(f.get("timestamp", 0.0)) >= target_ts]
    if later_frames:
        later_frames.sort(key=lambda x: float(x.get("timestamp", 0.0)))
        return later_frames[0]
    if not frames:
        return None
    frames.sort(key=lambda x: float(x.get("timestamp", 0.0)))
    return frames[-1]


def _refine_frames_with_secondary_retrieval(
    frames_info: List[Dict[str, Any]],
    subtitles: List[Dict[str, Any]],
    *,
    max_num_frames: int,
    subtitle_enabled: bool,
    subtitle_query: str,
    time_enabled: bool,
    time_range: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if max_num_frames <= 0:
        return [], []

    normalized_frames = _dedupe_and_normalize_frames(frames_info)
    normalized_frames.sort(key=lambda x: float(x.get("timestamp", 0.0)))

    selected_frames: List[Dict[str, Any]] = []
    matched_subtitle = None

    if subtitle_enabled:
        matched_subtitle = _find_best_subtitle_match(subtitles, subtitle_query)
        if matched_subtitle:
            start, end = matched_subtitle.get("timestamp", (0.0, 0.0))
            start = float(start)
            end = float(end)
            subtitle_ts = (start + end) / 2

            pre_start = max(0.0, start - 2.0)
            pre_end = max(pre_start, start)
            selected_frames.extend(
                _sample_frames_in_range(normalized_frames, pre_start, pre_end, count=2)
            )

            at_frame = _pick_frame_at_timestamp(normalized_frames, start)
            if at_frame:
                selected_frames.append(at_frame)

            post_end = subtitle_ts + 5.0
            selected_frames.extend(
                _sample_frames_in_range(
                    normalized_frames, subtitle_ts, post_end, count=5
                )
            )

    time_frames: List[Dict[str, Any]] = []
    if time_enabled and time_range:
        start = float(time_range.get("start_sec") or 0.0)
        end = float(time_range.get("end_sec") or start)
        if end < start:
            start, end = end, start
        sample_count = 4 if subtitle_enabled else 8
        time_frames = _sample_frames_in_range(normalized_frames, start, end, sample_count)
        selected_frames.extend(time_frames)

    selected_keys = set()
    unique_selected: List[Dict[str, Any]] = []
    for frame in selected_frames:
        key = (frame.get("output_path"), float(frame.get("timestamp", 0.0)))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        unique_selected.append(frame)

    remaining_budget = max_num_frames - len(unique_selected)
    if remaining_budget > 0:
        scored_frames = sorted(
            normalized_frames,
            key=lambda x: float(x.get("score") or 0.0),
            reverse=True,
        )
        for frame in scored_frames:
            key = (frame.get("output_path"), float(frame.get("timestamp", 0.0)))
            if key in selected_keys:
                continue
            unique_selected.append(frame)
            selected_keys.add(key)
            remaining_budget -= 1
            if remaining_budget <= 0:
                break

    if len(unique_selected) > max_num_frames:
        indices = np.linspace(0, len(unique_selected) - 1, max_num_frames, dtype=int)
        unique_selected = [unique_selected[i] for i in indices]

    unique_selected.sort(key=lambda x: float(x.get("timestamp", 0.0)))

    if not unique_selected:
        return [], []

    frame_timestamps = [float(item.get("timestamp", 0.0)) for item in unique_selected]
    filtered_subtitles: List[Dict[str, Any]] = []
    for entry in subtitles:
        start, end = entry.get("timestamp", (0.0, 0.0))
        start = float(start)
        end = float(end)
        center = (start + end) / 2
        if any(abs(center - ts) <= 0.5 for ts in frame_timestamps):
            filtered_subtitles.append(entry)
    if matched_subtitle and matched_subtitle not in filtered_subtitles:
        filtered_subtitles.append(matched_subtitle)
    filtered_subtitles.sort(key=lambda x: x.get("timestamp", (0.0, 0.0))[0])

    return unique_selected, filtered_subtitles


def _load_frames_from_results(paths: List[Dict[str, Any]], max_num_frames: int) -> Tuple[List[Image.Image], List[float]]:
    sorted_frames = sorted(paths, key=lambda x: float(x.get("timestamp") or 0.0))
    if max_num_frames > 0 and len(sorted_frames) > max_num_frames:
        indices = np.linspace(0, len(sorted_frames) - 1, max_num_frames, dtype=int)
        sorted_frames = [sorted_frames[i] for i in indices]

    frames: List[Image.Image] = []
    timestamps: List[float] = []
    for item in sorted_frames:
        frame_path = item.get("output_path") or item.get("path")
        if not frame_path:
            continue
        frame_path = os.path.abspath(frame_path)
        if not os.path.exists(frame_path):
            continue
        with open(frame_path, "rb") as f:
            frame = Image.open(f).convert("RGB")
        frames.append(frame)
        timestamps.append(float(item.get("timestamp") or 0.0))

    return frames, timestamps


class LongVideoBenchDataset(Dataset):
    def __init__(
        self,
        output_root,
        max_num_frames=256,
        insert_text=True,
        insert_frame=True,
    ):
        super().__init__()
        self.output_root = output_root
        self.insert_text = insert_text
        self.max_num_frames = max_num_frames

        self.data: List[Dict[str, Any]] = []
        batch_correct_choices = _load_batch_correct_choices(output_root)
        for name in sorted(os.listdir(output_root)):
            video_dir = os.path.join(output_root, name)
            if not os.path.isdir(video_dir):
                continue

            rerank_path = os.path.join(video_dir, "rerank_results.json")
            plan_path = os.path.join(video_dir, "retrieval_plan.json")
            time_focus_path = os.path.join(video_dir, "time_focus_results.json")

            if not os.path.exists(rerank_path) or not os.path.exists(plan_path):
                continue

            rerank_results = _load_json(rerank_path)
            retrieval_plan = _load_json(plan_path)
            time_focus_results = _load_json(time_focus_path) if os.path.exists(time_focus_path) else []

            correct_choice = _parse_choice_index(batch_correct_choices.get(name))
            if correct_choice is None:
                correct_choice = _parse_choice_index(retrieval_plan.get("correct_choice"))
            if correct_choice is None:
                correct_choice = -1

            selected_segments = retrieval_plan.get("selected_segments") or []
            subtitles, max_segment_end = _gather_subtitles(selected_segments)

            duration = float(retrieval_plan.get("video_metadata", {}).get("duration_sec") or max_segment_end)

            self.data.append(
                {
                    "id": name,
                    "rerank_results": rerank_results,
                    "time_focus_results": time_focus_results,
                    "subtitles": subtitles,
                    "duration": duration,
                    "intent": retrieval_plan.get("intent", {}),
                    "subtitle_analysis": retrieval_plan.get("subtitle_analysis", {}),
                    "query_variants": retrieval_plan.get("query_variants", {}),
                    "time_focus_range": retrieval_plan.get("time_focus_range", {}),
                    "question": retrieval_plan.get("query_variants", {}).get("original")
                    or retrieval_plan.get("query_variants", {}).get("vision")
                    or "",
                    "candidates": retrieval_plan.get("candidates", []),
                    "correct_choice": correct_choice,
                }
            )

    def __getitem__(self, index):
        di = self.data[index]

        def _format_correct_choice() -> str:
            choice_index = int(di.get("correct_choice", -1))
            if choice_index < 0:
                return "@"
            return chr(ord("A") + choice_index)

        if self.max_num_frames == 0:
            inputs = []
            inputs += ["Question: " + (di.get("question") or "")]
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di.get("candidates", []))]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": _format_correct_choice(), "id": di["id"]}

        frames_info = list(di.get("rerank_results", [])) + list(di.get("time_focus_results", []))

        intent = di.get("intent", {}) or {}
        subtitle_query = (
            (di.get("subtitle_analysis") or {}).get("subtitle_text")
            or (di.get("query_variants") or {}).get("subtitle")
            or ""
        )
        refined_frames_info, filtered_subtitles = _refine_frames_with_secondary_retrieval(
            frames_info,
            di.get("subtitles", []),
            max_num_frames=self.max_num_frames,
            subtitle_enabled=bool(intent.get("subtitle_search")),
            subtitle_query=subtitle_query,
            time_enabled=bool(intent.get("time_search")),
            time_range=di.get("time_focus_range"),
        )

        if refined_frames_info:
            frames, frame_timestamps = _load_frames_from_results(
                refined_frames_info, len(refined_frames_info)
            )
            subtitles = filtered_subtitles
        else:
            frames, frame_timestamps = _load_frames_from_results(
                frames_info, self.max_num_frames
            )
            subtitles = di.get("subtitles", [])
        inputs: List[Any] = []
        if self.insert_text:
            inputs = insert_subtitles_into_frames(
                frames,
                frame_timestamps,
                subtitles,
                starting_timestamp_for_subtitles=0.0,
                duration=di.get("duration", 0.0),
            )
        else:
            inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        if di.get("question"):
            inputs += ["Question: " + di["question"]]
        if di.get("candidates"):
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di["candidates"])]
        inputs += ["Answer with the option's letter from the given choices directly."]
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        return {"inputs": inputs, "correct_choice": _format_correct_choice(), "id": di["id"]}

    def __len__(self):
        return len(self.data)

    def get_id(self, index):
        return self.data[index]["id"]


if __name__ == "__main__":
    def _describe_inputs(inputs: List[Any]) -> List[str]:
        descriptions: List[str] = []
        for idx, item in enumerate(inputs):
            if isinstance(item, str):
                descriptions.append(f"{idx:02d} text: {item}")
            elif isinstance(item, Image.Image):
                descriptions.append(
                    f"{idx:02d} frame: size={item.size} mode={item.mode}"
                )
            else:
                descriptions.append(f"{idx:02d} unknown type: {type(item)}")
        return descriptions

    def _preview_dataset(db: "LongVideoBenchDataset", *, label: str):
        print(f"\n===== {label} =====")
        for i in range(min(3, len(db))):
            sample = db[i]
            print("\nSample ID:", sample["id"])
            frame_count = len(
                [ele for ele in sample["inputs"] if not isinstance(ele, str)]
            )
            print("Frame count:", frame_count)
            print("Detailed inputs:")
            for desc in _describe_inputs(sample["inputs"]):
                print(desc)

    # Default preview
    db = LongVideoBenchDataset("./output")
    _preview_dataset(db, label="Default (max_num_frames=256)")

    # Example showing a refined pipeline capped at 16 frames after secondary retrieval
    db_refined = LongVideoBenchDataset("./output", max_num_frames=16)
    _preview_dataset(db_refined, label="Refined (max_num_frames=16)")
                     

            
            