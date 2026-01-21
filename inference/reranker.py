from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import os
import shutil
import tempfile
from collections import defaultdict
from typing import List, Dict, Iterable, Optional
from peft import PeftModel
from time_utils import timestamp_label

adapter_dir = "./result"  # 👈 替换为你训练保存LoRA的目录

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

_tokenizer = processor.tokenizer
id_1 = _tokenizer.convert_tokens_to_ids("1")
id_2 = _tokenizer.convert_tokens_to_ids("2")
id_3 = _tokenizer.convert_tokens_to_ids("3")
id_4 = _tokenizer.convert_tokens_to_ids("4")
id_5 = _tokenizer.convert_tokens_to_ids("5")


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    id1_vector = batch_scores[:, id_1]
    id2_vector = batch_scores[:, id_2]
    id3_vector = batch_scores[:, id_3]
    id4_vector = batch_scores[:, id_4]
    id5_vector = batch_scores[:, id_5]
    batch_scores = torch.stack([id1_vector, id2_vector, id3_vector, id4_vector, id5_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores.exp().tolist()
    return scores


def format_message(query, image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
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
                    ),
                },
            ],
        }
    ]


def _score_segment_frames(
    segment_info: Dict,
    query: str,
    frame_interval: int,
    temp_dir: str,
    *,
    target_sample_fps: Optional[float] = None,
) -> List[Dict]:
    os.makedirs(temp_dir, exist_ok=True)
    video_path = segment_info["path"]
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(segment_info.get("fps", 0) or 0.0)
    if fps <= 0.0:
        fps = 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {video_path} | {total_frames} frames at {fps} FPS")

    frame_results: List[Dict] = []

    idx = 0
    success, frame = cap.read()
    effective_interval = max(int(frame_interval), 1)
    if target_sample_fps and target_sample_fps > 0.0 and fps > 0.0:
        computed = int(round(fps / float(target_sample_fps)))
        effective_interval = max(computed, 1)
    while success:
        if idx % effective_interval == 0:
            segment_index = segment_info.get("segment_index")
            if segment_index is not None:
                segment_index = int(segment_index)
            else:
                segment_index = 0

            frame_filename = f"seg{segment_index:04d}_frame_{idx:05d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            messages = format_message(query, frame_path)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            probs = compute_logits(inputs)[0]
            weighted_sum = sum((i + 1) * p for i, p in enumerate(probs))

            start_frame = segment_info.get("start_frame", 0)
            if start_frame is None:
                start_frame = 0
            global_frame_index = int(start_frame) + idx
            timestamp = global_frame_index / fps if fps else 0.0

            frame_results.append(
                {
                    "temp_path": frame_path,
                    "score": weighted_sum,
                    "segment_index": segment_index,
                    "segment_path": video_path,
                    "frame_in_segment": idx,
                    "global_frame_index": global_frame_index,
                    "timestamp": timestamp,
                }
            )

        success, frame = cap.read()
        idx += 1

    cap.release()
    print(f"Extracted {len(frame_results)} frames for scoring")

    return frame_results


def rerank_segments(
    segment_infos: Iterable[Dict],
    query: str,
    frame_interval: int = 10,
    top_frames: int = 128,
    output_dir: str = "reranker_output",
    min_frames_per_clip: int = 6,
    *,
    target_sample_fps: Optional[float] = None,
) -> List[Dict]:
    """对检索到的片段进行帧级重排序，并导出最相关的帧。

    在全局排序前，会优先保留每个片段中得分最高的 ``min_frames_per_clip`` 帧，
    以避免高分片段被完全过滤掉。"""

    temp_dir = tempfile.mkdtemp(prefix="frames_tmp_")
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_frames: List[Dict] = []
        for info in segment_infos:
            all_frames.extend(
                _score_segment_frames(
                    info,
                    query=query,
                    frame_interval=frame_interval,
                    temp_dir=temp_dir,
                    target_sample_fps=target_sample_fps,
                )
            )

        if not all_frames:
            return []

        if top_frames <= 0:
            return []

        grouped_frames: Dict[int, List[Dict]] = defaultdict(list)
        for frame_info in all_frames:
            grouped_frames[int(frame_info["segment_index"])].append(frame_info)

        for frame_list in grouped_frames.values():
            frame_list.sort(key=lambda x: x["score"], reverse=True)

        initial_selection: List[Dict] = []
        if min_frames_per_clip > 0:
            for frame_list in grouped_frames.values():
                initial_selection.extend(frame_list[:min_frames_per_clip])

        initial_selection.sort(key=lambda x: x["score"], reverse=True)
        if len(initial_selection) > top_frames:
            selected_frames = initial_selection[:top_frames]
        else:
            selected_frames = list(initial_selection)
            remaining_frames: List[Dict] = []
            for frame_list in grouped_frames.values():
                start_idx = min_frames_per_clip if min_frames_per_clip > 0 else 0
                remaining_frames.extend(frame_list[start_idx:])

            remaining_frames.sort(key=lambda x: x["score"], reverse=True)
            for frame_info in remaining_frames:
                if len(selected_frames) >= top_frames:
                    break
                selected_frames.append(frame_info)

        selected_frames = selected_frames[:top_frames]
        selected_frames.sort(
            key=lambda x: (x["timestamp"], x["segment_index"], x["frame_in_segment"])
        )

        results: List[Dict] = []
        for rank, frame_info in enumerate(selected_frames, start=1):
            timestamp = float(frame_info.get("timestamp") or 0.0)
            timestamp_tag = timestamp_label(timestamp)
            filename = (
                f"t{timestamp_tag}_"
                f"seg{frame_info['segment_index']:04d}_"
                f"frame{frame_info['frame_in_segment']:05d}.jpg"
            )
            dest_path = os.path.join(output_dir, filename)
            shutil.copy2(frame_info["temp_path"], dest_path)

            enriched = dict(frame_info)
            enriched["rank"] = rank
            enriched["output_path"] = dest_path
            enriched["timestamp_label"] = timestamp_tag
            results.append(enriched)

        return results

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Rerank frames for selected segments")
    parser.add_argument("segments", help="JSON file with segment infos")
    parser.add_argument("query", help="Text query")
    parser.add_argument("--frame-interval", type=int, default=10, dest="frame_interval")
    parser.add_argument("--top-frames", type=int, default=128, dest="top_frames")
    parser.add_argument("--output", default="reranker_output")

    args = parser.parse_args()

    with open(args.segments, "r", encoding="utf-8") as f:
        infos = json.load(f)

    results = rerank_segments(
        infos,
        query=args.query,
        frame_interval=args.frame_interval,
        top_frames=args.top_frames,
        output_dir=args.output,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
