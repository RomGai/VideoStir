import os
import json
import zipfile
import shutil
import cv2
from tqdm import tqdm

zip_path = "./[Update]_Anet_videos_15fps_short256.zip"
extract_root = "./Anet_partial_extract"

train_json_path = "./train.json"
test_json_path  = "./test.json"

PREFIX = "Anet_partial_extract"

tmp_video_dir = os.path.join(extract_root, "_tmp_videos")

video_exts = (".mp4", ".avi", ".mov", ".mkv")


def iter_image_paths_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for conversation in data:
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        p = item.get("image")
                        if isinstance(p, str):
                            yield p


def video_id_from_image_path(p, prefix=PREFIX):
    p = p.replace("\\", "/").lstrip("./")
    parts = [x for x in p.split("/") if x]
    if prefix in parts:
        i = parts.index(prefix)
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def collect_target_video_ids(*json_paths):
    vids = set()
    for jp in json_paths:
        for img_path in iter_image_paths_from_json(jp):
            vid = video_id_from_image_path(img_path)
            if vid:
                vids.add(vid)
    return vids


def safe_extract_one_member(zf, member, extract_to):
    extract_to = os.path.abspath(extract_to)
    dest = os.path.abspath(os.path.join(extract_to, member))
    if not (dest.startswith(extract_to + os.sep) or dest == extract_to):
        raise RuntimeError(f"Unsafe path detected in zip member: {member}")
    zf.extract(member, extract_to)
    return os.path.join(extract_to, member)


def sample_video_frames(video_path, output_dir, interval=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return 0

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved

def rmdir_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

os.makedirs(extract_root, exist_ok=True)
os.makedirs(tmp_video_dir, exist_ok=True)

target_vids = collect_target_video_ids(train_json_path, test_json_path)
print(f"Collected {len(target_vids)} target video ids.")

with zipfile.ZipFile(zip_path, "r") as zf:
    file_list = [m for m in zf.namelist() if m.lower().endswith(video_exts)]
    print(f"Found {len(file_list)} video files in zip.")

    base_to_members = {}
    for m in file_list:
        base = os.path.splitext(os.path.basename(m))[0]
        base_to_members.setdefault(base, []).append(m)

    members_to_process = []
    missing = []
    multi = []

    for vid in target_vids:
        ms = base_to_members.get(vid, [])
        if not ms:
            missing.append(vid)
        else:
            if len(ms) > 1:
                multi.append((vid, ms))
            members_to_process.extend(ms)

    members_to_process = sorted(set(members_to_process))

    print(f"Matched members to process: {len(members_to_process)}")
    if missing:
        print(f"Missing (no matching video file in zip): {len(missing)}")
        print("Example missing:", missing[:10])
    if multi:
        print(f"Warning: {len(multi)} video ids have multiple matches (will process all).")
        print("Example multi:", multi[0])

    results = []
    for member in tqdm(members_to_process, desc="Extract->Sample->Delete"):
        base_name = os.path.splitext(os.path.basename(member))[0]

        out_dir = os.path.join(extract_root, base_name)

        if os.path.isdir(out_dir):
            existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")]
            if len(existing) > 0:
                results.append((base_name, "skipped", len(existing)))
                continue

        extracted_path = safe_extract_one_member(zf, member, tmp_video_dir)

        saved = sample_video_frames(extracted_path, out_dir, interval=5)

        try:
            os.remove(extracted_path)
        except FileNotFoundError:
            pass

        cur = os.path.dirname(extracted_path)
        while os.path.abspath(cur).startswith(os.path.abspath(tmp_video_dir)):
            if os.path.isdir(cur) and not os.listdir(cur):
                os.rmdir(cur)
                cur = os.path.dirname(cur)
            else:
                break

        results.append((base_name, "done", saved))

rmdir_if_exists(tmp_video_dir)

print("All done. Temp video dir removed:", tmp_video_dir)

done = sum(1 for _, status, _ in results if status == "done")
skipped = sum(1 for _, status, _ in results if status == "skipped")
print(f"Done videos: {done}, skipped videos: {skipped}")
print("Sample results:")
for r in results[:10]:
    print(r)
