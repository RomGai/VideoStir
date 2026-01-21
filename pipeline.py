import argparse
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from split import extract_visual_embeddings, cluster_and_segment, export_segments
from chunk_embedding import compute_video_features
from build_graph import build_spatiotemporal_graph
from topk_graph_retrieval import retrieve_topk_segments
from reranker import rerank_segments
from query_intent import (
    analyze_query_intent,
    analyze_time_focus,
    rewrite_query_and_extract_subtitles,
)
from text_embedding import compute_similarities
from time_utils import seconds_to_timestamp, timestamp_label


_NEAR_ZERO_DURATION = 1e-3


def _to_serializable(obj: Any) -> Any:
    """Recursively convert objects containing numpy/torch types for JSON dumping."""

    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(item) for item in obj]

    if isinstance(obj, (np.generic,)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    return obj

def _normalize_subtitle_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return None
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + part
            return seconds
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _resolve_path(base_dir: Optional[str], candidate: Optional[str]) -> Optional[str]:
    """Resolve ``candidate`` relative to ``base_dir`` if it's not absolute."""

    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    if os.path.isabs(candidate):
        return candidate
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    base_dir = base_dir or ""
    return os.path.abspath(os.path.join(base_dir, candidate))


def _probe_video_metadata(video_path: str) -> Dict[str, float]:
    import cv2

    metadata = {"fps": 0.0, "total_frames": 0.0, "duration": 0.0}
    if not video_path:
        return metadata

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return metadata

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        duration = total_frames / fps if fps > 0.0 else 0.0
        metadata.update({"fps": fps, "total_frames": total_frames, "duration": duration})
        return metadata
    finally:
        cap.release()


def _load_subtitle_entries(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[pipeline] Subtitle file not found, skipping: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        candidates = raw.get("subtitles") or raw.get("segments") or []
    else:
        candidates = raw

    entries: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        start = _normalize_subtitle_time(item.get("start") or item.get("start_time"))
        end = _normalize_subtitle_time(item.get("end") or item.get("end_time"))
        text = item.get("text") or item.get("content") or item.get("line") or ""
        if start is None or end is None:
            continue
        entries.append({"start": float(start), "end": float(end), "text": str(text)})

    entries.sort(key=lambda x: x["start"])
    return entries


def _attach_subtitles(
    segment_infos: Iterable[Dict[str, Any]], subtitle_entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    subtitle_entries = list(subtitle_entries)
    enriched_segments: List[Dict[str, Any]] = []
    for info in segment_infos:
        start = float(info.get("start_sec", 0.0) or 0.0)
        end = float(info.get("end_sec", start))
        matched_texts: List[str] = []
        matched_entries: List[Dict[str, Any]] = []
        for entry in subtitle_entries:
            entry_start = float(entry.get("start", 0.0) or 0.0)
            entry_end = float(entry.get("end", entry_start) or entry_start)
            if entry_end < entry_start:
                entry_start, entry_end = entry_end, entry_start

            duration = entry_end - entry_start
            if duration <= _NEAR_ZERO_DURATION:
                if entry_start < start:
                    continue
                if entry_start > end:
                    break
            else:
                if entry_end < start:
                    continue
                if entry_start > end:
                    break
            matched_texts.append(entry["text"])
            matched_entries.append(entry)

        enriched = dict(info)
        enriched["subtitle_text"] = " ".join(t for t in matched_texts if t).strip()
        enriched["subtitle_entries"] = matched_entries
        enriched_segments.append(enriched)

    return enriched_segments


def _time_attribute_text(attrs: Dict[str, Any]) -> str:
    start = attrs.get("start_sec")
    end = attrs.get("end_sec")
    if start is None or end is None:
        return ""
    timestamp_start = seconds_to_timestamp(start)
    timestamp_end = seconds_to_timestamp(end)
    duration = max(float(end) - float(start), 0.0)
    return (
        f"Clip spanning {timestamp_start} to {timestamp_end} (duration {duration:.2f} seconds)."
    )


def _subtitle_attribute_text(attrs: Dict[str, Any]) -> str:
    text = attrs.get("subtitle_text")
    if not text:
        return ""
    return str(text)


def _retrieve_nodes_by_attribute(
    graph, query: str, text_getter, top_k: int
) -> List[Tuple[int, float, str]]:
    documents: List[str] = []
    node_ids: List[int] = []
    for node_id, attrs in graph.nodes(data=True):
        text = text_getter(attrs)
        if not text:
            continue
        node_ids.append(node_id)
        documents.append(text)

    if not documents:
        return []

    stripped_query = (query or "").strip()

    # Prefer exact/character-level matches of the subtitle text inside the source subtitle
    # entries. This helps avoid cases where semantic similarity misses verbatim subtitle
    # snippets provided by the user.
    if stripped_query:
        normalized_query = stripped_query.lower()
        substring_hits: List[Tuple[int, float, str]] = []
        for node_id, doc in zip(node_ids, documents):
            if normalized_query in doc.lower():
                substring_hits.append((node_id, 1.0, doc))

        if substring_hits:
            return substring_hits[: max(top_k, 1)]

    sims = compute_similarities(query, documents)
    scored = list(zip(node_ids, sims, documents))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(top_k, 1)]


def _combine_segment_scores(info: Dict[str, Any]) -> float:
    vision_score = float(info.get("similarity") or 0.0)
    subtitle_score = float(info.get("subtitle_similarity") or 0.0)
    time_score = float(info.get("time_similarity") or 0.0)
    combined = vision_score + subtitle_score + time_score
    info["combined_score"] = combined
    return combined

def _collect_temporal_neighbors(graph, node_id: int, hops: int) -> List[int]:
    if hops <= 0:
        return []

    visited = {node_id}
    frontier = {node_id}
    collected = set()

    for _ in range(hops):
        next_frontier = set()
        for current in frontier:
            for neighbor in graph.neighbors(current):
                edge = graph.edges[current, neighbor]
                if edge.get("type") != "temporal":
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                collected.add(neighbor)
                next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier

    return list(collected)


def _merge_attribute_results(
    graph,
    base_segments: Iterable[Dict[str, Any]],
    intent: Dict[str, Any],
    query: str,
    attribute_top_k: int,
    subtitle_query: Optional[str] = None,
    subtitle_neighbor_hops: int = 3,
) -> List[Dict[str, Any]]:
    aggregated = {info["node_id"]: dict(info) for info in base_segments if "node_id" in info}

    for info in aggregated.values():
        info.setdefault("subtitle_similarity", 0.0)
        info.setdefault("time_similarity", 0.0)

    subtitle_hits: List[Tuple[int, float, str]] = []
    if intent.get("subtitle_search"):
        subtitle_query_text = subtitle_query if subtitle_query else query
        subtitle_hits = _retrieve_nodes_by_attribute(
            graph, subtitle_query_text, _subtitle_attribute_text, attribute_top_k
        )
        print("------------subtitle_hits:")
        print(subtitle_hits)
        for node_id, sim, _ in subtitle_hits:
            info = aggregated.get(node_id)
            if info is None:
                attrs = dict(graph.nodes[node_id])
                attrs["node_id"] = node_id
                attrs["similarity"] = 0.0
                attrs["subtitle_similarity"] = 0.0
                attrs["time_similarity"] = 0.0
                aggregated[node_id] = attrs
                info = attrs
            info["subtitle_similarity"] = float(sim)

    if intent.get("subtitle_search") and subtitle_neighbor_hops > 0:
        subtitle_hit_scores = {node_id: float(sim) for node_id, sim, _ in subtitle_hits}
        for node_id, sim in subtitle_hit_scores.items():
            temporal_neighbors = _collect_temporal_neighbors(
                graph, node_id, subtitle_neighbor_hops
            )
            for neighbor_id in temporal_neighbors:
                info = aggregated.get(neighbor_id)
                if info is None:
                    attrs = dict(graph.nodes[neighbor_id])
                    attrs["node_id"] = neighbor_id
                    attrs["similarity"] = 0.0
                    attrs["subtitle_similarity"] = 0.0
                    attrs["time_similarity"] = 0.0
                    aggregated[neighbor_id] = attrs
                    info = attrs
                current = float(info.get("subtitle_similarity") or 0.0)
                if sim > current:
                    info["subtitle_similarity"] = sim

    merged = list(aggregated.values())
    for info in merged:
        _combine_segment_scores(info)

    merged.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return merged


def _sparse_sample_time_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    frame_interval: int,
    output_dir: str,
    prefix: str,
) -> List[Dict[str, Any]]:
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0.0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(int(start_sec * fps), 0)
    end_frame = min(int(end_sec * fps), total_frames - 1 if total_frames > 0 else start_frame)
    if end_frame < start_frame:
        end_frame = start_frame

    frame_interval = max(int(frame_interval), 1)

    sampled: List[Dict[str, Any]] = []
    current = start_frame
    while current <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        success, frame = cap.read()
        if not success:
            break

        timestamp = current / fps if fps else 0.0
        filename = f"t{timestamp_label(timestamp)}_{prefix}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

        sampled.append(
            {
                "frame_index": current,
                "timestamp": timestamp,
                "output_path": output_path,
            }
        )

        current += frame_interval

    cap.release()
    return sampled


def run_batch_from_config(
    config_path: str,
    output_root: str,
    *,
    video_root: Optional[str] = None,
    subtitle_root: Optional[str] = None,
    default_query_field: str = "query_retrieval",
    fallback_query_fields: Optional[Iterable[str]] = None,
    skip_existing: bool = True,
    **pipeline_kwargs,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the pipeline for every entry defined in a batch configuration file.

    Args:
        config_path: Path to the JSON file describing the batch inputs.
        output_root: Directory where per-video results will be written.
        video_root: Optional base directory for resolving video paths.
        subtitle_root: Optional base directory for resolving subtitle paths.
        default_query_field: Preferred key in each JSON entry for the query text.
        fallback_query_fields: Additional keys to look up if the default is missing.
        **pipeline_kwargs: Additional keyword arguments passed to :func:`run_pipeline`.
        skip_existing: When ``True`` (default), skip entries whose output directory
            already exists under ``output_root``. This allows interrupted runs to
            resume without repeating completed work. When ``False``, existing
            outputs are overwritten.

    Returns:
        A mapping from entry identifiers (``id`` when available, otherwise the
        video stem) to the resulting data returned by :func:`run_pipeline`.
    """

    with open(config_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Batch configuration must be a JSON list of entries.")

    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    batch_results: Dict[str, Dict[str, List[Dict]]] = {}
    fallback_fields = list(fallback_query_fields or ["query", "query_vlm"])

    def _with_metadata(results: Dict[str, Any], entry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attach entry-level metadata (e.g., correct_choice) to the result payload."""

        if "correct_choice" not in entry_data:
            return results

        merged = dict(results)
        merged["correct_choice"] = entry_data.get("correct_choice")
        return merged

    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            print(f"[pipeline] Skipping non-dict entry at index {idx - 1}.")
            continue

        video_rel_path = entry.get("video_path")
        video_path = _resolve_path(video_root, video_rel_path)
        if not video_path or not os.path.exists(video_path):
            print(
                f"[pipeline] Skipping entry {entry.get('id', idx)}: video not found"
                f" ({video_rel_path})."
            )
            continue

        subtitle_rel_path = entry.get("subtitle_path")
        subtitle_path = _resolve_path(subtitle_root, subtitle_rel_path)
        if subtitle_path and not os.path.exists(subtitle_path):
            subtitle_path = None

        query = entry.get(default_query_field)
        if not query:
            for field in fallback_fields:
                query = entry.get(field)
                if query:
                    break
        if not query:
            print(f"[pipeline] Skipping entry {entry.get('id', idx)}: missing query text.")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        item_id = entry.get("id") or video_name
        video_output_dir = os.path.join(output_root, video_name)

        if skip_existing and os.path.isdir(video_output_dir):
            print(
                f"[pipeline] Skipping entry {idx}/{len(entries)} (id={item_id}): "
                f"existing output found at {video_output_dir}."
            )

            existing_results: Dict[str, List[Dict]] = {}
            rerank_path = os.path.join(video_output_dir, "rerank_results.json")
            time_path = os.path.join(video_output_dir, "time_focus_results.json")

            if os.path.exists(rerank_path):
                try:
                    with open(rerank_path, "r", encoding="utf-8") as f:
                        existing_results["reranked_frames"] = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    print(
                        f"[pipeline] Warning: Failed to load existing rerank results "
                        f"for {item_id}: {exc}"
                    )

            if os.path.exists(time_path):
                try:
                    with open(time_path, "r", encoding="utf-8") as f:
                        existing_results["time_focus_frames"] = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    print(
                        f"[pipeline] Warning: Failed to load existing time focus results "
                        f"for {item_id}: {exc}"
                    )

            if existing_results:
                batch_results[item_id] = _with_metadata(existing_results, entry)
            else:
                batch_results[item_id] = _with_metadata({"skipped": True}, entry)
            continue

        print("=" * 80)
        print(
            f"[pipeline] Processing entry {idx}/{len(entries)}: id={item_id}, video={video_rel_path}"
        )

        try:
            results = run_pipeline(
                video_path=video_path,
                query=query,
                output_dir=video_output_dir,
                subtitle_json=subtitle_path,
                correct_choice=entry.get("correct_choice"),
                **pipeline_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[pipeline] Skipping entry {idx}/{len(entries)} (id={item_id}) due to error: {exc}"
            )
            if os.path.isdir(video_output_dir):
                try:
                    shutil.rmtree(video_output_dir)
                except OSError as cleanup_exc:
                    print(
                        f"[pipeline] Warning: Failed to clean output directory for {item_id}: {cleanup_exc}"
                    )

            batch_results[item_id] = _with_metadata({"skipped": True, "error": str(exc)}, entry)
            continue

        batch_results[item_id] = _with_metadata(results, entry)

    summary_path = os.path.join(output_root, "batch_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(batch_results), f, ensure_ascii=False, indent=2)

    print(f"[pipeline] Batch processing complete. Summary saved to {summary_path}")

    return batch_results


def run_pipeline(
    video_path: str,
    query: str,
    output_dir: str,
    frame_interval: int = 30,
    n_clusters: int = 10,
    min_segment_sec: float = 0.4,
    embedding_frame_interval: int = 10,
    top_k: int = 3,
    spatial_k: int = 3,
    rerank_frame_interval: int = 5,
    top_frames: int = 128,
    temporal_weight: float = 1.0,
    subtitle_json: Optional[str] = None,
    attribute_top_k: int = 3,
    min_frames_per_clip: int = 6,
    subtitle_neighbor_hops: int = 2,
    time_focus_ratio: float = 0.05,
    time_sampling_interval: int = 10,
    time_range_padding: float = 1.0,
    time_min_window: float = 2.0,
    short_video_threshold: float = 240.0,
    correct_choice: Optional[Any] = None,
) -> Dict[str, List[Dict]]:
    """执行完整的长视频检索与重排序流程。

    当视频时长低于 ``short_video_threshold`` 时，会自动跳过视觉语义检索，
    直接以 3 FPS 采样整段视频并调用重排模块进行排序。

    Returns:
        dict: 包含 ``reranked_frames``（经过节点检索和重排的帧列表）以及
        ``time_focus_frames``（基于时间范围稀疏采样得到的帧列表）的字典。
        当传入 ``correct_choice`` 时，会附加到生成的检索计划中。
    """

    temp_root = tempfile.mkdtemp(prefix="pipeline_tmp_")
    segments_dir = os.path.join(temp_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    output_dir = os.path.abspath(output_dir)
    # 确保最终输出目录不在临时工作区中，避免清理临时目录时误删最终结果
    if os.path.commonpath([output_dir, temp_root]) == temp_root:
        raise ValueError(
            "Output directory must be outside the pipeline's temporary workspace."
        )

    os.makedirs(output_dir, exist_ok=True)

    try:
        video_metadata = _probe_video_metadata(video_path)
        video_fps = float(video_metadata.get("fps", 0.0) or 0.0)
        video_total_frames = float(video_metadata.get("total_frames", 0.0) or 0.0)
        approx_duration = float(video_metadata.get("duration", 0.0) or 0.0)
        short_video_mode = (
            short_video_threshold > 0.0
            and approx_duration > 0.0
            and approx_duration <= short_video_threshold
        )

        if short_video_mode:
            print(
                f"[pipeline] Short video detected (duration {approx_duration:.2f}s). "
                "Skipping spatiotemporal retrieval and sampling at 3 FPS."
            )

        subtitle_entries = _load_subtitle_entries(subtitle_json)

        if short_video_mode:
            end_frame_index = int(video_total_frames) - 1 if video_total_frames > 0.0 else 0
            segment_infos = [
                {
                    "segment_index": 0,
                    "path": video_path,
                    "start_sec": 0.0,
                    "end_sec": approx_duration if approx_duration > 0.0 else 0.0,
                    "start_frame": 0,
                    "end_frame": max(end_frame_index, 0),
                    "fps": video_fps if video_fps > 0.0 else None,
                }
            ]
            if subtitle_entries:
                segment_infos = _attach_subtitles(segment_infos, subtitle_entries)
            else:
                segment_infos = list(segment_infos)
            graph = None
        else:
            embeddings, frames = extract_visual_embeddings(
                video_path, frame_interval=frame_interval
            )
            change_points, _ = cluster_and_segment(
                video_path,
                embeddings,
                frames,
                method="kmeans",
                n_clusters=n_clusters,
            )

            segment_infos = export_segments(
                video_path,
                change_points,
                output_dir=segments_dir,
                min_segment_sec=min_segment_sec,
            )

            if not segment_infos:
                raise RuntimeError("No segments were generated from the input video.")

            if subtitle_entries:
                segment_infos = _attach_subtitles(segment_infos, subtitle_entries)
            else:
                segment_infos = list(segment_infos)

            graph = build_spatiotemporal_graph(
                compute_video_features(segment_infos, frame_interval=embedding_frame_interval),
                temporal_weight=temporal_weight,
            )

        print("--------------------segment_infos:")
        print(segment_infos)

        original_query = query.strip()
        intent = analyze_query_intent(query)
        print(
            "Query intent:",
            json.dumps({
                "subtitle_search": intent.get("subtitle_search"),
                "time_search": intent.get("time_search"),
                "reason": intent.get("reason"),
            }, ensure_ascii=False),
        )

        subtitle_query_text: Optional[str] = None
        cleaned_query = original_query or query
        subtitle_analysis: Optional[Dict[str, Any]] = None
        time_focus_analysis: Optional[Dict[str, Any]] = None
        time_focus_results: List[Dict[str, Any]] = []

        if intent.get("subtitle_search"):
            subtitle_analysis = rewrite_query_and_extract_subtitles(query)
            extracted_subtitle = subtitle_analysis.get("subtitle_text") if subtitle_analysis else ""
            print("----------extracted_subtitles:")
            print(extracted_subtitle)
            if extracted_subtitle:
                subtitle_query_text = extracted_subtitle

                cleaned_candidate = (
                    subtitle_analysis.get("cleaned_query") if subtitle_analysis else ""
                )
                if cleaned_candidate:
                    cleaned_query = cleaned_candidate
            else:
                # The intent model requested subtitle search, but no subtitle text was
                # extracted from the query. Fall back to the original query and skip
                # subtitle retrieval for this run.
                intent["subtitle_search"] = False
                subtitle_query_text = None
                cleaned_query = original_query

            print(
                "Subtitle rewrite:",
                json.dumps(
                    {
                        "subtitle_text": subtitle_query_text or "",
                        "cleaned_query": cleaned_query,
                        "reason": subtitle_analysis.get("reason") if subtitle_analysis else "",
                    },
                    ensure_ascii=False,
                ),
            )

        vision_query = cleaned_query.strip() or original_query or query

        print("----------query with no subtitles:")
        print(vision_query)

        total_duration = max((info.get("end_sec") or 0.0) for info in segment_infos)
        if total_duration > 0.0:
            approx_duration = total_duration

        if intent.get("time_search"):
            time_focus_analysis = analyze_time_focus(query)
            print(
                "Time focus analysis:",
                json.dumps(
                    {
                        "mode": time_focus_analysis.get("mode"),
                        "start_time_sec": time_focus_analysis.get("start_time_sec"),
                        "end_time_sec": time_focus_analysis.get("end_time_sec"),
                        "reason": time_focus_analysis.get("reason"),
                    },
                    ensure_ascii=False,
                ),
            )

            mode = (time_focus_analysis.get("mode") or "none").lower()
            start_sec = float(time_focus_analysis.get("start_time_sec") or 0.0)
            end_sec = float(time_focus_analysis.get("end_time_sec") or 0.0)

            clamped_ratio = max(min(float(time_focus_ratio), 1.0), 0.0)
            if mode == "start":
                end_sec = total_duration * clamped_ratio
                start_sec = 0.0
            elif mode == "end":
                start_sec = max(total_duration * (1.0 - clamped_ratio), 0.0)
                end_sec = total_duration
            elif mode == "range":
                start_sec = max(min(start_sec, total_duration), 0.0)
                end_sec = max(min(end_sec, total_duration), 0.0)
            else:
                start_sec = 0.0
                end_sec = 0.0

            if end_sec < start_sec:
                start_sec, end_sec = end_sec, start_sec

            if end_sec - start_sec < max(float(time_min_window), 0.0):
                mid = (start_sec + end_sec) / 2.0
                padding = max(float(time_range_padding), 0.0)
                half_window = max(float(time_min_window) / 2.0, padding)
                start_sec = max(mid - half_window, 0.0)
                end_sec = min(mid + half_window, total_duration)

            if end_sec > start_sec:
                time_output_dir = os.path.join(output_dir, "time_focus_frames")
                mode_label = mode if mode in {"start", "end", "range"} else "custom"
                prefix = f"time_{mode_label}"
                time_focus_results = _sparse_sample_time_range(
                    video_path,
                    start_sec,
                    end_sec,
                    frame_interval=time_sampling_interval,
                    output_dir=time_output_dir,
                    prefix=prefix,
                )

                for item in time_focus_results:
                    item["mode"] = mode_label
                    item["start_sec"] = start_sec
                    item["end_sec"] = end_sec
                    item["timestamp_label"] = timestamp_label(item.get("timestamp", 0.0))

        if short_video_mode:
            selected_segments = list(segment_infos)
            rerank_inputs = list(segment_infos)
        else:
            effective_top_k = top_k
            effective_spatial_k = spatial_k
            if intent.get("subtitle_search"):
                effective_top_k = 1
                effective_spatial_k = 1

            selected_segments = retrieve_topk_segments(
                graph,
                vision_query,
                top_k=effective_top_k,
                spatial_k=effective_spatial_k,
            )

            if not selected_segments:
                raise RuntimeError("No segments were selected from the retrieval stage.")

            merged_segments = _merge_attribute_results(
                graph,
                selected_segments,
                intent=intent,
                query=vision_query,
                attribute_top_k=attribute_top_k,
                subtitle_query=subtitle_query_text,
                subtitle_neighbor_hops=subtitle_neighbor_hops,
            )

            max_segments = max(top_k + attribute_top_k, len(selected_segments))
            rerank_inputs = merged_segments[:max_segments]

        if not rerank_inputs:
            raise RuntimeError("No segments are available for reranking.")

        selected_segments = list(rerank_inputs)

        final_frames = rerank_segments(
            rerank_inputs,
            query=vision_query,
            frame_interval=rerank_frame_interval,
            top_frames=top_frames,
            output_dir=output_dir,
            min_frames_per_clip=min_frames_per_clip,
            target_sample_fps=3.0 if short_video_mode else None,
        )

        results_path = os.path.join(output_dir, "rerank_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(final_frames), f, ensure_ascii=False, indent=2)

        if time_focus_results:
            time_results_path = os.path.join(output_dir, "time_focus_results.json")
            with open(time_results_path, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(time_focus_results), f, ensure_ascii=False, indent=2)

        plan_path = os.path.join(output_dir, "retrieval_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "intent": intent,
                    "selected_segments": _to_serializable(selected_segments),
                    "query_variants": {
                        "original": query,
                        "vision": vision_query,
                        "subtitle": subtitle_query_text or "",
                    },
                    "subtitle_analysis": subtitle_analysis,
                    "time_focus_analysis": time_focus_analysis,
                    "time_focus_range": {
                        "start_sec": time_focus_results[0]["start_sec"] if time_focus_results else 0.0,
                        "end_sec": time_focus_results[0]["end_sec"] if time_focus_results else 0.0,
                    },
                    "short_video_mode": short_video_mode,
                    "video_metadata": {
                        "fps": video_fps,
                        "total_frames": video_total_frames,
                        "duration_sec": approx_duration,
                    },
                    "correct_choice": correct_choice,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "reranked_frames": final_frames,
            "time_focus_frames": time_focus_results,
        }

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full long-video retrieval pipeline")
    # parser.add_argument("--video", type=str, default="./videos/f44gpGR4uWU.mp4", help="Path to the input video")
    # parser.add_argument("--query", type=str, default="Question: In the opening of the video, there's a man wearing a black top and a gray hat in the car. In which of the following scenes does he appear later? A. In the water. B. In the car, on the sofa. C. On the mountain. D. In the bathroom.", help="Text query for retrieval")
    # parser.add_argument("--output", type=str, default="./output/", help="Directory to save the final ranked frames")
    parser.add_argument("--video", type=str, default=None, help="Path to the input video")
    parser.add_argument("--query", type=str, default=None, help="Text query for retrieval")
    parser.add_argument("--output", type=str, default="./output/", help="Directory to save the final ranked frames")
    parser.add_argument("--frame-interval", type=int, default=45, dest="frame_interval")
    parser.add_argument("--clusters", type=int, default=30, dest="n_clusters")
    parser.add_argument("--min-segment-sec", type=float, default=1, dest="min_segment_sec")
    parser.add_argument("--embed-frame-interval", type=int, default=10, dest="embedding_frame_interval")
    parser.add_argument("--top-k", type=int, default=1, dest="top_k")
    parser.add_argument("--spatial-k", type=int, default=1, dest="spatial_k")
    parser.add_argument("--rerank-frame-interval", type=int, default=5, dest="rerank_frame_interval")
    parser.add_argument("--top-frames", type=int, default=256, dest="top_frames")
    parser.add_argument("--temporal-weight", type=float, default=1.0, dest="temporal_weight")
    # parser.add_argument("--subtitle-json", type=str, default="./subtitles/f44gpGR4uWU_en.json", dest="subtitle_json")
    parser.add_argument("--subtitle-json", type=str, default=None, dest="subtitle_json")
    parser.add_argument("--attribute-top-k", type=int, default=1, dest="attribute_top_k")
    parser.add_argument("--min-frames-per-clip", type=int, default=4, dest="min_frames_per_clip")
    parser.add_argument(
        "--subtitle-neighbor-hops", type=int, default=3, dest="subtitle_neighbor_hops"
    )
    parser.add_argument(
        "--time-focus-ratio", type=float, default=0.04, dest="time_focus_ratio"
    )
    parser.add_argument(
        "--time-sampling-interval", type=int, default=20, dest="time_sampling_interval"
    )
    parser.add_argument(
        "--time-range-padding", type=float, default=1.0, dest="time_range_padding"
    )
    parser.add_argument(
        "--time-min-window", type=float, default=2.0, dest="time_min_window"
    )
    parser.add_argument(
        "--short-video-threshold",
        type=float,
        default=240.0,
        dest="short_video_threshold",
        help="Duration (in seconds) below which the pipeline skips graph retrieval and samples at 3 FPS.",
    )
    parser.add_argument("--batch-config", type=str, default="sampled_longvideobench_val_augmented.json", dest="batch_config", help="JSON file describing batch inputs")
    parser.add_argument("--video-root", type=str, default="./videos", dest="video_root", help="Base directory for resolving relative video paths in batch mode")
    parser.add_argument(
        "--subtitle-root",
        type=str,
        default="./subtitles",
        dest="subtitle_root",
        help="Base directory for resolving relative subtitle paths in batch mode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess entries even if their output directory already exists. "
            "By default, the batch runner skips completed videos so that runs can resume."
        ),
    )

    args = parser.parse_args()

    pipeline_kwargs = dict(
        frame_interval=args.frame_interval,
        n_clusters=args.n_clusters,
        min_segment_sec=args.min_segment_sec,
        embedding_frame_interval=args.embedding_frame_interval,
        top_k=args.top_k,
        spatial_k=args.spatial_k,
        rerank_frame_interval=args.rerank_frame_interval,
        top_frames=args.top_frames,
        temporal_weight=args.temporal_weight,
        attribute_top_k=args.attribute_top_k,
        min_frames_per_clip=args.min_frames_per_clip,
        subtitle_neighbor_hops=args.subtitle_neighbor_hops,
        time_focus_ratio=args.time_focus_ratio,
        time_sampling_interval=args.time_sampling_interval,
        time_range_padding=args.time_range_padding,
        time_min_window=args.time_min_window,
        short_video_threshold=args.short_video_threshold,
    )

    batch_config = args.batch_config.strip()
    if batch_config:
        run_batch_from_config(
            config_path=batch_config,
            output_root=args.output,
            video_root=args.video_root,
            subtitle_root=args.subtitle_root,
            skip_existing=not args.force,
            **pipeline_kwargs,
        )
    else:
        video_path = _resolve_path(args.video_root, args.video)
        subtitle_path = _resolve_path(args.subtitle_root, args.subtitle_json)

        results = run_pipeline(
            video_path=video_path,
            query=args.query,
            output_dir=args.output,
            subtitle_json=subtitle_path,
            **pipeline_kwargs,
        )

        print(json.dumps(results, ensure_ascii=False, indent=2))
