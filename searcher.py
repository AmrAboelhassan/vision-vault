"""Search a local video-frame CLIP database with natural language queries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "openai/clip-vit-base-patch32"
COLLECTION_NAME = "video_vault"
DISTRACTOR_LABELS = (
    "a floor",
    "a wall",
    "a ceiling",
    "blurry movement",
    "empty space",
)


def load_runtime_dependencies() -> None:
    global chromadb, cv2, torch, Image, CLIPModel, CLIPProcessor

    import chromadb
    import cv2
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed video frames using natural language."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search phrase, for example: red shirt. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("security_db"),
        help="Directory containing the ChromaDB database.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("search_results"),
        help="Directory where matching frames are exported.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of nearest frames to inspect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum verification confidence required to export a frame.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the first exported match in an OpenCV preview window.",
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
        print("Using CPU. Search will be slower.")

    model.eval()
    return model, processor


def sanitize_filename(text: str) -> str:
    name = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    return name or "query"


def query_embedding(
    query: str,
    *,
    model: Any,
    processor: Any,
    device: Any,
) -> list[float]:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().float().numpy().flatten().tolist()


def verify_frame(
    image_path: Path,
    query: str,
    *,
    model: Any,
    processor: Any,
    device: Any,
) -> tuple[float, object | None]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return 0.0, None

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels = [f"a photo of {query}", *DISTRACTOR_LABELS]
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    if device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)

    return probabilities[0][0].item(), frame


def export_matches(
    query: str,
    *,
    collection,
    model: Any,
    processor: Any,
    device: Any,
    output_path: Path,
    top_k: int,
    threshold: float,
) -> list[Path]:
    record_count = collection.count()
    if record_count == 0:
        print("The database is empty. Run indexer.py first.")
        return []

    output_dir = output_path / sanitize_filename(query)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching for: {query}")
    print(f"Inspecting top {min(top_k, record_count)} of {record_count:,} indexed frames.")
    print(f"Exporting matches to: {output_dir}")

    results = collection.query(
        query_embeddings=[
            query_embedding(query, model=model, processor=processor, device=device)
        ],
        n_results=min(top_k, record_count),
    )

    exported_paths: list[Path] = []
    for metadata in results["metadatas"][0]:
        image_path = Path(metadata["path"])
        if not image_path.is_file():
            continue

        score, frame = verify_frame(
            image_path,
            query,
            model=model,
            processor=processor,
            device=device,
        )
        if frame is None or score < threshold:
            continue

        timestamp = metadata["time"].replace(":", "-")
        camera = sanitize_filename(metadata["camera"])
        export_path = output_dir / f"{camera}_{timestamp}_score_{int(score * 100)}.jpg"
        cv2.imwrite(str(export_path), frame)
        exported_paths.append(export_path)

    print(f"Found {len(exported_paths)} high-confidence matches.")
    return exported_paths


def preview_first_match(paths: list[Path], query: str) -> None:
    if not paths:
        return

    frame = cv2.imread(str(paths[0]))
    if frame is None:
        return

    cv2.imshow(f"Preview: {query}", cv2.resize(frame, (640, 360)))
    print("Press any key in the preview window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_search(args: argparse.Namespace, collection, model, processor, device) -> None:
    print(f"\nAI Search Console ({collection.count():,} indexed records)")
    print("Type a search phrase, or 'q' to quit.")

    while True:
        query = input("\nSearch query: ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        matches = export_matches(
            query,
            collection=collection,
            model=model,
            processor=processor,
            device=device,
            output_path=args.output_path,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        if args.preview:
            preview_first_match(matches, query)


def main() -> int:
    args = parse_args()
    load_runtime_dependencies()

    device = resolve_device(args.device)
    model, processor = load_clip(device)

    try:
        client = chromadb.PersistentClient(path=str(args.db_path))
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"Could not open database at '{args.db_path}'. Run indexer.py first.")
        print(f"Details: {exc}")
        return 1

    if args.query:
        query = " ".join(args.query)
        matches = export_matches(
            query,
            collection=collection,
            model=model,
            processor=processor,
            device=device,
            output_path=args.output_path,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        if args.preview:
            preview_first_match(matches, query)
        return 0

    interactive_search(args, collection, model, processor, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
