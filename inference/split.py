import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os


def _lazy_import_ruptures():
    """Lazy import ruptures to avoid hard dependency at import time."""

    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            "The 'ruptures' package is required for PELT-based segmentation. "
            "Install it with `pip install ruptures`."
        ) from exc
    return rpt

visual_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
vis_emb_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    visual_model_id, attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained(visual_model_id)
vis_emb_model.visual.to("cuda").eval()

def extract_visual_embeddings(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames, embeddings = [], []
    idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor.image_processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to("cuda")
                grid_thw = inputs["image_grid_thw"].to("cuda")
                vision_outputs = vis_emb_model.visual(pixel_values, grid_thw)
                emb = vision_outputs.mean(dim=0).squeeze(0).cpu().numpy()  # shape [2048]
                embeddings.append(emb)
                frames.append(idx)
                print(idx)
            idx += 1
    cap.release()
    return np.array(embeddings), np.array(frames)

def detect_change_points_pelt(
    embeddings, frames, penalty=5.0, min_size=5, jump=1, cost_model="rbf"
):

    if len(embeddings) != len(frames):
        raise ValueError("The lengths of the embeddings and the frames do not match.")
    if len(embeddings) < 2:
        return [int(frames[0]), int(frames[-1])]

    pairwise_sim = cosine_similarity(embeddings)
    semantic_shift = 1.0 - np.clip(np.diag(pairwise_sim, k=1), 0.0, 1.0)
    series = np.concatenate([[0.0], semantic_shift])[:, None]

    rpt = _lazy_import_ruptures()
    algo = rpt.Pelt(model=cost_model, jump=jump, min_size=min_size).fit(series)
    idx_change_points = algo.predict(pen=penalty)

    change_points = [int(frames[0])]
    for idx in idx_change_points:
        if idx >= len(frames):
            continue
        change_points.append(int(frames[idx]))

    if change_points[-1] != int(frames[-1]):
        change_points.append(int(frames[-1]))

    change_points = sorted(dict.fromkeys(change_points))
    return change_points

def cluster_and_segment(video_path, embeddings, frames, method="kmeans", n_clusters=5):
    print("clusting")
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=1.5, min_samples=3)
        labels = clusterer.fit_predict(embeddings)
    elif method == "pelt":
        change_points = detect_change_points_pelt(embeddings, frames)
        labels = np.zeros(len(frames), dtype=int)
        for seg_id in range(1, len(change_points)):
            start_frame = change_points[seg_id - 1]
            end_frame = change_points[seg_id]
            mask = (frames >= start_frame) & (frames < end_frame)
            labels[mask] = seg_id - 1
    else:
        raise ValueError("Unsupported clustering method")

    if method != "pelt":
        change_points = [frames[0]]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                change_points.append(frames[i])
        change_points.append(frames[-1])
    return change_points, labels

def export_segments(
    video_path,
    change_points,
    output_prefix="segment",
    min_segment_sec=10,
    output_dir="segments",
):
    """Export video segments and return their metadata.
    
    If a segment is shorter than the threshold (in seconds), it will be merged into the next segment.
    Consecutive short segments will be merged together into the following longer segment, and the
    output files will be saved to the specified folder.
    
    Args:
        video_path (str): Path to the video.
        change_points (list[int]): A list of frame indices where cuts occur.
        output_prefix (str): Prefix for output filenames.
        min_segment_sec (float): Minimum segment duration (seconds). Segments shorter than this will be
            merged into the next segment.
        output_dir (str): Output directory path (created automatically if it does not exist).
    
    Returns:
        list[dict]: Metadata for each segment, including ``path``, ``start_frame``, ``end_frame``, and ``fps``.
    """

    print("Export video segments")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    min_frames = int(fps * min_segment_sec)

    merged_points = [change_points[0]]
    i = 1
    while i < len(change_points):
        start, end = change_points[i - 1], change_points[i]
        seg_len = end - start

        if seg_len < min_frames and i < len(change_points) - 1:
            i += 1
            continue
        else:
            merged_points.append(change_points[i])
            i += 1

    segment_infos = []
    seg_id = 0
    for i in range(len(merged_points) - 1):
        start, end = merged_points[i], merged_points[i + 1]
        out_path = os.path.join(output_dir, f"{output_prefix}_{seg_id}.mp4")

        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for j in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"🎬 Saved: {out_path}")
        start_sec = float(start) / fps if fps else 0.0
        end_sec = float(end) / fps if fps else start_sec
        segment_infos.append(
            {
                "path": out_path,
                "start_frame": start,
                "end_frame": end,
                "fps": fps,
                "segment_index": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(end_sec - start_sec, 0.0),
            }
        )
        seg_id += 1

    cap.release()
    print(f"Exported {seg_id} segments to folder: {os.path.abspath(output_dir)}")

    return segment_infos

if __name__ == "__main__":
    video_path = "Demo.mp4"
    embeddings, frames = extract_visual_embeddings(video_path, frame_interval=5)
    change_points, labels = cluster_and_segment(video_path, embeddings, frames, method="kmeans", n_clusters=10)
    export_segments(video_path, change_points)

