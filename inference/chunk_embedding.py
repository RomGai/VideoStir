import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import os
from typing import Iterable, List, Dict
import torch
from PIL import Image
import decord

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'PE-Core-G14-448'
model = pe.CLIP.from_config(model_name, pretrained=True).to(device)

preprocess = transforms.get_image_transform(model.image_size)

def preprocess_video(video_path, frame_interval=5, transform=None):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    frame_indices = list(range(0, total_frames, frame_interval))
    frames = vr.get_batch(frame_indices).asnumpy()
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    return torch.stack(preprocessed_frames, dim=0)

# # ====== 批处理整个文件夹 ======
# def compute_video_features(
#     segment_infos: Iterable[Dict],
#     frame_interval: int = 10,
# ) -> List[Dict]:
#     """为给定的视频片段计算视觉特征。

#     Args:
#         segment_infos: 包含 ``path`` 键的片段信息迭代器。
#         frame_interval: 抽帧间隔。

#     Returns:
#         list[dict]: 每个元素包含原始 ``segment_info``，并额外附带 ``image_features``。
#     """

#     results: List[Dict] = []

#     with torch.no_grad():
#         for info in segment_infos:
#             video_path = info["path"]
#             if not os.path.exists(video_path):
#                 print(f"[x] Missing file: {video_path}")
#                 continue

#             try:
#                 video = preprocess_video(video_path, frame_interval=frame_interval, transform=preprocess)
#                 video = video.unsqueeze(0).to(device)

#                 image_features = model.encode_video(video)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)

#                 enriched = dict(info)
#                 enriched["image_features"] = image_features.cpu()
#                 results.append(enriched)
#                 print(f"[✓] Processed {os.path.basename(video_path)}")

#             except Exception as e:
#                 print(f"[x] Failed on {video_path}: {e}")

#     return results

def compute_video_features(
    segment_infos: Iterable[Dict],
    frame_interval: int = 10,
    chunk_size: int = 16,   # 每批处理帧数，可调大/小
) -> List[Dict]:
    """为给定的视频片段计算视觉特征（显存安全版，支持分块执行）。

    Args:
        segment_infos: 包含 ``path`` 键的片段信息迭代器。
        frame_interval: 抽帧间隔。
        chunk_size: 每次送入模型的帧数（显存敏感参数，默认16）。

    Returns:
        list[dict]: 每个元素包含原始 ``segment_info``，并额外附带 ``image_features``。
    """

    results: List[Dict] = []

    with torch.no_grad():
        for info in segment_infos:
            video_path = info["path"]
            if not os.path.exists(video_path):
                print(f"[x] Missing file: {video_path}")
                continue

            try:
                # ---- 预处理（仍在 CPU）----
                video = preprocess_video(video_path, frame_interval=frame_interval, transform=preprocess)
                total_frames = video.shape[0]
                if total_frames == 0:
                    print(f"[x] No valid frames in {video_path}")
                    continue

                # ---- 分块执行 ----
                feats = []
                for i in range(0, total_frames, chunk_size):
                    chunk = video[i:i + chunk_size].unsqueeze(0).to(device, non_blocking=True)
                    f = model.encode_video(chunk)
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.cpu())

                    # 显存清理
                    del chunk, f
                    torch.cuda.empty_cache()

                # ---- 聚合所有块特征 ----
                # 你原逻辑是 image_features.shape=[1,1280]
                # 这里对所有 chunk 取平均
                image_features = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)

                enriched = dict(info)
                enriched["image_features"] = image_features
                results.append(enriched)

                print(f"[✓] Processed {os.path.basename(video_path)} ({total_frames} frames, chunk={chunk_size})")

                del video, feats, image_features
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[x] Failed on {video_path}: {e}")
                torch.cuda.empty_cache()
                continue

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute CLIP features for video segments.")
    parser.add_argument("segment_info", help="JSON file containing a list of segment metadata.")
    parser.add_argument("output", help="Path to save the resulting features (pt file).")
    parser.add_argument("--frame-interval", type=int, default=5, dest="frame_interval")

    args = parser.parse_args()

    with open(args.segment_info, "r", encoding="utf-8") as f:
        infos = json.load(f)

    results = compute_video_features(infos, frame_interval=args.frame_interval)
    torch.save(results, args.output)
    print(f"Saved {len(results)} segments to {args.output}")
