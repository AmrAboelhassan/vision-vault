"""Build a searchable CLIP embedding database from video files."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "openai/clip-vit-base-patch32"
COLLECTION_NAME = "video_vault"


def load_runtime_dependencies() -> None:
    global chromadb, cv2, torch, Image, CLIPModel, CLIPProcessor

    import chromadb
    import cv2
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index video frames into a local ChromaDB database using CLIP embeddings."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Video files to index. If omitted, paths are requested interactively.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("security_db"),
        help="Directory where ChromaDB stores the vector database.",
    )
    parser.add_argument(
        "--frames-path",
        type=Path,
        default=Path("db_images"),
        help="Directory where sampled video frames are saved.",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=2.0,
        help="Seconds between indexed frames.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device to use.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable GPU was found.")

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(requested)


def load_clip(device: Any) -> tuple[Any, Any]:
    print(f"Loading CLIP model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    if device.type == "cuda":
        model = model.half()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. Indexing will be slower.")

    model.eval()
    return model, processor


def prompt_for_videos() -> list[Path]:
    videos: list[Path] = []
    print("\nEnter video paths one at a time. Type 'done' when finished.")

    while True:
        raw_path = input(f"Video path #{len(videos) + 1}: ").strip().strip('"')
        if raw_path.lower() == "done":
            break

        path = Path(raw_path)
        if path.is_file():
            videos.append(path)
            print(f"Added: {path.name}")
        else:
            print(f"File not found: {path}")

    return videos


def validate_videos(videos: Iterable[Path]) -> list[Path]:
    valid_videos = []

    for video in videos:
        if video.is_file():
            valid_videos.append(video)
        else:
            print(f"Skipping missing file: {video}")

    return valid_videos


def frame_embedding(
    frame_bgr,
    *,
    model: Any,
    processor: Any,
    device: Any,
) -> list[float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt").to(device)

    if device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().float().numpy().flatten().tolist()


def index_video(
    video_path: Path,
    *,
    collection,
    model: Any,
    processor: Any,
    device: Any,
    frames_path: Path,
    sample_every: float,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video: {video_path}")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_interval = max(1, int(fps * sample_every))
    indexed_count = 0
    frame_number = 0

    print(f"\nIndexing {video_path.name} ({total_frames:,} frames)")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_number % sample_interval == 0:
            timestamp_seconds = int(frame_number / fps)
            timestamp = str(dt.timedelta(seconds=timestamp_seconds))
            image_id = f"{video_path.stem}_{frame_number}"
            image_path = frames_path / f"{image_id}.jpg"

            embedding = frame_embedding(
                frame,
                model=model,
                processor=processor,
                device=device,
            )
            cv2.imwrite(str(image_path), frame)

            collection.upsert(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[
                    {
                        "camera": video_path.name,
                        "time": timestamp,
                        "path": str(image_path),
                        "frame": frame_number,
                    }
                ],
            )

            indexed_count += 1
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:5.1f}% | Indexed frame {frame_number}", end="\r")

        frame_number += 1

    capture.release()
    print(f"\nIndexed {indexed_count:,} frames from {video_path.name}")
    return indexed_count


def main() -> int:
    args = parse_args()
    load_runtime_dependencies()

    videos = validate_videos(args.videos) if args.videos else prompt_for_videos()

    if not videos:
        print("No valid videos provided.")
        return 1

    args.frames_path.mkdir(parents=True, exist_ok=True)
    args.db_path.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, processor = load_clip(device)

    client = chromadb.PersistentClient(path=str(args.db_path))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_indexed = 0
    for video in videos:
        total_indexed += index_video(
            video,
            collection=collection,
            model=model,
            processor=processor,
            device=device,
            frames_path=args.frames_path,
            sample_every=args.sample_every,
        )

    print(f"\nDone. Indexed {total_indexed:,} frames into '{args.db_path}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
