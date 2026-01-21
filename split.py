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

# ========== 初始化视觉模型 ==========
visual_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
vis_emb_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    visual_model_id, attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained(visual_model_id)
vis_emb_model.visual.to("cuda").eval()

# ========== 提取帧向量 ==========
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

# ========== PELT 变化点检测 ==========
def detect_change_points_pelt(
    embeddings, frames, penalty=5.0, min_size=5, jump=1, cost_model="rbf"
):
    """利用 PELT 对相邻帧语义相似度序列做自适应切割。

    Args:
        embeddings (np.ndarray): 形状为 ``[N, D]`` 的帧向量。
        frames (np.ndarray): 对应的帧编号，长度为 ``N``。
        penalty (float): PELT 的惩罚系数，越大切点越少。
        min_size (int): 邻接片段的最小样本数，防止过度切分。
        jump (int): PELT 扫描步长，可用于加速。
        cost_model (str): ruptures 成本函数，默认 ``rbf`` 适合非线性变化。

    Returns:
        list[int]: 以帧编号表示的切点列表（包含首尾）。
    """

    if len(embeddings) != len(frames):
        raise ValueError("embeddings 与 frames 长度不一致")
    if len(embeddings) < 2:
        return [int(frames[0]), int(frames[-1])]

    # 1. 构建“语义相似度变化”序列：相邻帧余弦相似度的反向值
    pairwise_sim = cosine_similarity(embeddings)
    # 只关注相邻帧，越大说明变化越剧烈
    semantic_shift = 1.0 - np.clip(np.diag(pairwise_sim, k=1), 0.0, 1.0)
    # 对齐长度，首元素补 0 代表起点
    series = np.concatenate([[0.0], semantic_shift])[:, None]

    rpt = _lazy_import_ruptures()
    algo = rpt.Pelt(model=cost_model, jump=jump, min_size=min_size).fit(series)
    # ruptures 返回的切点是索引+1 的位置（含结尾），需要映射回帧号
    idx_change_points = algo.predict(pen=penalty)

    change_points = [int(frames[0])]
    for idx in idx_change_points:
        # 跳过超出范围的末尾点，保证有序且唯一
        if idx >= len(frames):
            continue
        change_points.append(int(frames[idx]))

    if change_points[-1] != int(frames[-1]):
        change_points.append(int(frames[-1]))

    # 去重并排序，保证稳定
    change_points = sorted(dict.fromkeys(change_points))
    return change_points

# ========== 聚类并切割 ==========
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
        # 根据切点生成伪标签，便于后续可视化
        labels = np.zeros(len(frames), dtype=int)
        for seg_id in range(1, len(change_points)):
            start_frame = change_points[seg_id - 1]
            end_frame = change_points[seg_id]
            mask = (frames >= start_frame) & (frames < end_frame)
            labels[mask] = seg_id - 1
    else:
        raise ValueError("Unsupported clustering method")

    if method != "pelt":
        # 依据聚类结果找到切点
        change_points = [frames[0]]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                change_points.append(frames[i])
        change_points.append(frames[-1])
    return change_points, labels

# # ========== 导出子视频 ==========
# def export_segments(video_path, change_points, output_prefix="segment"):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#
#     seg_id = 0
#     for i in range(len(change_points) - 1):
#         start, end = change_points[i], change_points[i + 1]
#         out = cv2.VideoWriter(f"{output_prefix}_{seg_id}.mp4", fourcc, fps, (w, h))
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start)
#         for j in range(start, end):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
#         out.release()
#         seg_id += 1
#     cap.release()
#     print(f"✅ Exported {seg_id} segments.")


def export_segments(
    video_path,
    change_points,
    output_prefix="segment",
    min_segment_sec=10,
    output_dir="segments",
):
    """导出视频片段，并返回片段的元信息。

    若单个片段小于阈值（秒），则合并到下一个片段。连续短片段会整体
    合并到后一个长片段中，并将输出保存到指定文件夹中。

    Args:
        video_path (str): 视频路径。
        change_points (list[int]): 切割帧号列表。
        output_prefix (str): 输出文件名前缀。
        min_segment_sec (float): 最小片段时长（秒），小于该值的会合并到下一个片段。
        output_dir (str): 输出文件夹路径（若不存在将自动创建）。

    Returns:
        list[dict]: 每个片段的元信息，包含 ``path``、``start_frame``、``end_frame`` 和 ``fps``。
    """
    print("导出视频")

    # ===== 创建输出文件夹 =====
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    min_frames = int(fps * min_segment_sec)

    # ===== 合并短片段的切点逻辑 =====
    merged_points = [change_points[0]]
    i = 1
    while i < len(change_points):
        start, end = change_points[i - 1], change_points[i]
        seg_len = end - start

        if seg_len < min_frames and i < len(change_points) - 1:
            # 当前段太短 → 合并到下一个段
            i += 1
            continue
        else:
            merged_points.append(change_points[i])
            i += 1

    # ===== 导出视频阶段 =====
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
    print(f"✅ Exported {seg_id} segments to folder: {os.path.abspath(output_dir)}")

    return segment_infos


# ========== 主流程 ==========
if __name__ == "__main__":
    video_path = "Demo_19s.mp4"
    embeddings, frames = extract_visual_embeddings(video_path, frame_interval=5)
    change_points, labels = cluster_and_segment(video_path, embeddings, frames, method="kmeans", n_clusters=10)
    export_segments(video_path, change_points)
