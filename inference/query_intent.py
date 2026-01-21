"""Utilities for analyzing query intent with a Qwen VL model."""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_processor: AutoProcessor | None = None
_intent_model: Qwen2_5_VLForConditionalGeneration | None = None


def _load_model() -> tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    global _processor, _intent_model

    if _processor is None or _intent_model is None:
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _intent_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        _intent_model.eval()

    return _processor, _intent_model


def _unload_model() -> None:
    """Release cached processor/model references and clear GPU memory."""

    global _processor, _intent_model

    if _processor is None and _intent_model is None:
        return

    model = _intent_model
    processor = _processor

    _intent_model = None
    _processor = None

    # Drop references before forcing garbage collection / cache cleanup.
    del processor
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gc.collect()


INTENT_PROMPT = (
    "You are assisting with long-video retrieval. Analyze the user's query to determine "
    "whether the system should activate additional subtitle-based retrieval and/or time-range retrieval.\n"
    "Return a strict JSON object with the following keys: 'subtitle_search' (true or false), "
    "'time_search' (true or false), and 'reason' (a brief sentence explaining your decision).\n"
    "Focus only on explicit or strongly implied cues in the query.\n"
    "For example, if the query mentions a specific time segment of the video (e.g., 'opening', 'beginning', 'end'), "
    "enable time_search; otherwise, set it to false.\n"
    "If the query refers to a particular subtitle or quoted line, enable subtitle_search; otherwise, set it to false.\n"
    "Query: {query}\n"
    "JSON:"
)

SUBTITLE_REWRITE_PROMPT = (
    "You assist with long-video question answering. The user query may contain quoted "
    "or paraphrased subtitles mixed with other instructions. Extract only the subtitle "
    "text that should be matched against a subtitle index, and rewrite the query so "
    "that it no longer contains any literal subtitle-related text (for example: \"After the subtitle '......'\"; \"When the phrase '.....'\") while keeping all other "
    "statements and the final question.\n"
    "Return a strict JSON object with keys: 'subtitle_text' (a single string with the "
    "subtitle text separated by spaces, or an empty string if none), 'cleaned_query' "
    "(the query rewritten without subtitle-related text but preserving the rest), and 'reason' "
    "(briefly explain your extraction).\n"
    "Query: {query}\n"
    "JSON:"
)

TIME_FOCUS_PROMPT = (
    "You help a video-retrieval pipeline understand time-oriented requests.\n"
    "Given the query, decide whether it refers to the beginning, the ending, or a specific time span of the video.\n"
    "Return a JSON object with keys: 'mode', 'start_time_sec', 'end_time_sec', and 'reason'.\n"
    "'mode' must be one of: 'start', 'end', 'range', or 'none'.\n"
    "For queries about the start/opening/first moments, return mode 'start'.\n"
    "For queries about the end/closing/final moments, return mode 'end'.\n"
    "If the query specifies concrete timestamps (e.g., 'at 1:23', 'between 00:30 and 01:10'), return mode 'range' and fill "
    "'start_time_sec' and 'end_time_sec' with the interpreted seconds (use the same value for both if only one point is given).\n"
    "If no clear temporal focus exists, return mode 'none'.\n"
    "Always include 'reason' explaining the interpretation.\n"
    "Query: {query}\n"
    "JSON:"
)


def _generate_response(
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    unload_after_use: bool = True,
) -> str:
    processor, intent_model = _load_model()

    try:
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(text=[chat_text], return_tensors="pt").to(
            intent_model.device
        )
        with torch.no_grad():
            generated = intent_model.generate(**inputs, max_new_tokens=max_new_tokens)

        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    finally:
        if unload_after_use:
            intent_model = None
            processor = None
            _unload_model()


def _extract_json_object(response: str) -> Dict[str, Any]:
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return False


def analyze_query_intent(query: str, *, keep_model_loaded: bool = False) -> Dict[str, Any]:
    """Infer whether subtitle or temporal retrieval is required for a query.

    Args:
        query: Natural-language query to inspect.
        keep_model_loaded: Leave the underlying model cached after the call for
            subsequent invocations. Defaults to ``False`` to minimize GPU memory
            usage.
    """

    formatted_prompt = INTENT_PROMPT.format(query=query.strip())
    messages = [
        {"role": "system", "content": "You decide retrieval strategies for video search."},
        {"role": "user", "content": formatted_prompt},
    ]

    response = _generate_response(
        messages, unload_after_use=not keep_model_loaded
    )
    payload = _extract_json_object(response)

    subtitle_needed = _to_bool(payload.get("subtitle_search"))
    time_needed = _to_bool(payload.get("time_search"))
    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "subtitle_search": subtitle_needed,
        "time_search": time_needed,
        "reason": reason,
        "raw_response": response.strip(),
    }


def rewrite_query_and_extract_subtitles(
    query: str, *, keep_model_loaded: bool = False
) -> Dict[str, Any]:
    """Use the VL model to split subtitle text from the rest of the query.

    Args:
        query: User question potentially containing subtitle snippets.
        keep_model_loaded: Leave the model cached after the call for reuse.
            Defaults to ``False`` so the model is unloaded to free GPU memory.
    """

    formatted_prompt = SUBTITLE_REWRITE_PROMPT.format(query=query.strip())
    messages = [
        {
            "role": "system",
            "content": "You extract subtitle text and rewrite queries for video retrieval.",
        },
        {"role": "user", "content": formatted_prompt},
    ]

    response = _generate_response(
        messages,
        max_new_tokens=384,
        unload_after_use=not keep_model_loaded,
    )
    payload = _extract_json_object(response)

    subtitle_text = payload.get("subtitle_text")
    if isinstance(subtitle_text, str):
        subtitle_text = subtitle_text.strip()
    else:
        subtitle_text = ""

    cleaned_query = payload.get("cleaned_query")
    if isinstance(cleaned_query, str):
        cleaned_query = cleaned_query.strip()
    else:
        cleaned_query = ""

    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "subtitle_text": subtitle_text,
        "cleaned_query": cleaned_query,
        "reason": reason,
        "raw_response": response.strip(),
    }


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def analyze_time_focus(query: str, *, keep_model_loaded: bool = False) -> Dict[str, Any]:
    """Use the VL model to categorize the temporal focus of a query."""

    formatted_prompt = TIME_FOCUS_PROMPT.format(query=query.strip())
    messages = [
        {
            "role": "system",
            "content": "You interpret temporal hints in video-retrieval queries.",
        },
        {"role": "user", "content": formatted_prompt},
    ]

    response = _generate_response(
        messages,
        max_new_tokens=256,
        unload_after_use=not keep_model_loaded,
    )
    payload = _extract_json_object(response)

    mode = payload.get("mode")
    if isinstance(mode, str):
        mode = mode.strip().lower()
    else:
        mode = "none"

    start_time_sec = _to_float(payload.get("start_time_sec"))
    end_time_sec = _to_float(payload.get("end_time_sec"))
    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""

    return {
        "mode": mode,
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "reason": reason,
        "raw_response": response.strip(),
    }


def unload_intent_model() -> None:
    """Manually clear any cached model to release GPU memory."""

    _unload_model()

