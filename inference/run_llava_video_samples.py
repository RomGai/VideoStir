"""Run LongVideoBench samples with LLaVA-Video-7B-Qwen2.

This script mirrors ``run_refined_samples.py`` but targets the LLaVA-Video
model family. It builds prompts that respect the interleaved frame + subtitle
order produced by ``LongVideoBenchDataset`` and sends the resulting video
context to the model for reasoning.
"""

import copy
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from decord import cpu, VideoReader  # noqa: F401 (kept for parity with reference snippet)


def _ensure_llava_compatibility() -> None:
    """Backfill symbols expected by llava when using newer transformers.

    Some llava distributions import ``apply_chunking_to_forward`` from
    ``transformers.modeling_utils``. Newer versions of ``transformers`` have
    moved this helper, causing an ImportError during llava import. We shim the
    missing attribute to keep the script runnable without forcing a global
    downgrade.
    """

    try:
        from transformers import modeling_utils
    except Exception:
        return

    if hasattr(modeling_utils, "apply_chunking_to_forward"):
        return

    try:
        from transformers.pytorch_utils import apply_chunking_to_forward
    except Exception:
        return

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward


_ensure_llava_compatibility()

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token

from longvideobench_dataset import LongVideoBenchDataset

PRETRAINED_MODEL = os.getenv("LLAVA_PRETRAINED", "lmms-lab/LLaVA-Video-7B-Qwen2")
MODEL_NAME = os.getenv("LLAVA_MODEL_NAME", "llava_qwen")
DEVICE = os.getenv("LLAVA_DEVICE", "cuda")
DEVICE_MAP = os.getenv("LLAVA_DEVICE_MAP", "auto")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
CONV_TEMPLATE = os.getenv("LLAVA_CONV_TEMPLATE", "qwen_1_5")
RESULT_LOG_PATH = os.getenv(
    "LLAVA_RESULT_LOG", os.path.join("output", "llava_video_samples.json")
)


def _split_inputs(
    inputs: Sequence[Any],
) -> Tuple[List[np.ndarray], List[str], str, List[str]]:
    """Separate frames, subtitles, question, and candidates while keeping order."""

    frames: List[np.ndarray] = []
    timeline: List[str] = []
    question: str = ""
    candidates: List[str] = []

    for item in inputs:
        if isinstance(item, str):
            if item.startswith("Question: "):
                question = item[len("Question: ") :].strip()
            elif item.strip().lower().startswith("answer with"):
                # Terminal instruction; omit from timeline.
                continue
            elif len(item) >= 3 and item[1:3] == ". ":
                candidates.append(item.strip())
            else:
                timeline.append(f"Subtitle: {item.strip()}")
        else:
            frames.append(np.array(item))
            timeline.append(f"Frame {len(frames)}")

    return frames, timeline, question, candidates


def _build_prompt(timeline: List[str], question: str, candidates: List[str], frame_count: int) -> str:
    timeline_text = "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(timeline))

    option_text = "" if not candidates else "\nOptions:\n" + "\n".join(candidates)

    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"The following visual input contains {frame_count} uniformly sampled frames.\n"
        "Subtitles are interleaved in the timeline below to preserve their order relative to the frames.\n"
        f"Timeline:\n{timeline_text}\n"
        f"Question: {question}{option_text}\n"
        "Answer with the option's letter if choices are provided."
    )


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.bfloat16


def _get_video_target_dtype(model: torch.nn.Module) -> torch.dtype:
    """Prefer the projector's dtype when aligning vision inputs."""

    try:
        projector = model.get_model().mm_projector
        return next(projector.parameters()).dtype
    except Exception:
        return _get_model_dtype(model)


def _prepare_video_tensor(
    image_processor, frames: List[np.ndarray], device: str, dtype: torch.dtype
) -> torch.Tensor:
    if not frames:
        frames = [np.zeros((336, 336, 3), dtype=np.uint8)]

    video = image_processor.preprocess(np.stack(frames, axis=0), return_tensors="pt")[
        "pixel_values"
    ]
    return video.to(device=device, dtype=dtype)


def _run_sample(
    tokenizer, model, image_processor, sample: Dict[str, Any], device: str
) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))

    model_dtype = _get_model_dtype(model)
    video_target_dtype = _get_video_target_dtype(model)
    video = _prepare_video_tensor(image_processor, frames, device, video_target_dtype)
    print(
        "Video dtype before alignment: "
        f"{video.dtype}; model dtype: {model_dtype}; projector dtype: {video_target_dtype}"
    )
    if video.dtype != video_target_dtype:
        video = video.to(device=device, dtype=video_target_dtype)
        print(f"Converted video to {video_target_dtype} to match projector dtype.")
    images = [video]

    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=images,
        modalities=["video"],
        do_sample=False,
        max_new_tokens=512,
    )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def _append_sample_result(log_path: str, idx: int, sample: Dict[str, Any], reply: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    entry = {
        "sample_index": idx,
        "id": sample.get("id"),
        "correct_choice": sample.get("correct_choice", "@"),
        "model_reply": reply,
    }

    existing: List[Dict[str, Any]] = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, list):
                    existing = loaded
        except json.JSONDecodeError:
            existing = []

    existing.append(entry)

    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(existing, fp, ensure_ascii=False, indent=2)


def run_samples() -> None:
    tokenizer, model, image_processor, _ = load_pretrained_model(
        PRETRAINED_MODEL,
        None,
        MODEL_NAME,
        torch_dtype="bsfloat16",
        device_map=DEVICE_MAP,
    )
    model.eval()

    model = model.to(torch.bfloat16)
    model_dtype=_get_model_dtype(model)
    print(f"model dtype: {model_dtype}")

    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(tokenizer, model, image_processor, sample, DEVICE)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
