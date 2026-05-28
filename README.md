# Vision Vault

**AI-powered local video search using CLIP embeddings and ChromaDB.**

Vision Vault is a local-first computer vision prototype for searching video footage with natural-language queries.
It samples frames from video files, converts those frames into CLIP image embeddings, stores them in a local ChromaDB vector database, and exports matching frames when a text query is searched.

## Overview

Traditional video review can be slow when you need to find a specific object, scene, or visual condition across long footage.
Vision Vault makes that workflow searchable by combining semantic image embeddings with vector search.

Example use cases:

- Find frames containing a specific object, such as `red backpack` or `computer monitor`.
- Review long videos by searching for visual descriptions instead of manually scrubbing.
- Build a local prototype for semantic video search, retrieval, and frame export.

## Features

- Indexes one or more local video files.
- Samples frames at a configurable interval.
- Generates CLIP image embeddings for sampled frames.
- Stores embeddings and frame metadata in ChromaDB.
- Searches footage with natural-language text queries.
- Verifies candidate frames with CLIP image/text scoring.
- Exports high-confidence matching frames to organized folders.
- Supports CUDA acceleration when available.
- Keeps videos, generated frames, databases, and output folders out of Git.

## Architecture

```text
Video
  -> Sample Frames
  -> CLIP Image Embeddings
  -> ChromaDB
  -> Text Query
  -> CLIP Text Embedding
  -> Similarity Search
  -> Export Matching Frames
```

The project has two main scripts:

- `indexer.py` builds the local searchable frame database.
- `searcher.py` searches the database and exports matching frames.

## Demo

Demo screenshots will be added soon.

Suggested future asset structure:

```text
assets/
  demo-search-results.png
  architecture.png
```

No demo screenshots are included yet because generated or fake assets should not be used for portfolio presentation.

## Tech Stack

- Python
- PyTorch
- Transformers
- OpenAI CLIP model (`openai/clip-vit-base-patch32`)
- ChromaDB
- OpenCV
- Pillow

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The default `requirements.txt` installs PyTorch CUDA 12.1 wheels.
If you do not have a CUDA-compatible GPU, install the CPU PyTorch build from the official PyTorch instructions, then install the remaining dependencies.

## Usage

Index a video:

```powershell
python indexer.py "path\to\video.mp4"
```

Index multiple videos:

```powershell
python indexer.py "camera_1.mp4" "camera_2.mp4" --sample-every 2
```

Search indexed footage:

```powershell
python searcher.py "red shirt"
```

Run interactive search mode:

```powershell
python searcher.py
```

Preview the first exported match:

```powershell
python searcher.py "red shirt" --preview
```

View all options:

```powershell
python indexer.py --help
python searcher.py --help
```

## Example Queries

- `red shirt`
- `person carrying a backpack`
- `computer monitor`
- `white car`
- `person near the entrance`
- `empty hallway`
- `phone on a desk`

## Example Output

Search results are exported to a query-specific folder:

```text
search_results/
  red_shirt/
    camera_1_0-01-24_score_91.jpg
    camera_1_0-02-10_score_88.jpg
```

Each exported filename includes:

- the source video or camera name
- the timestamp from the video
- the confidence score used during CLIP verification

Generated output folders are ignored by Git so private footage and derived frames are not uploaded accidentally.

## Project Structure

```text
vision-vault/
  README.md
  LICENSE
  indexer.py
  searcher.py
  requirements.txt
  .gitignore
  .gitattributes
```

Local generated folders are intentionally excluded:

```text
db_images/
security_db/
search_results/
rtx_search_results/
trk_search_results/
```

## Safety / Privacy

Vision Vault is a local video search prototype.
It does not perform face recognition, does not identify people, and does not make security decisions.

Users should only process footage they own or have permission to use.
Video files, extracted frames, local databases, and search outputs can contain sensitive information, so they are ignored by Git by default.

## Limitations

- Results depend on CLIP's visual understanding and may include false positives or missed matches.
- This prototype searches sampled frames, not every frame, unless configured with a very small sampling interval.
- Large videos can require significant disk space for extracted frames.
- GPU acceleration is recommended for practical indexing speed.
- The system exports matching frames, but it does not provide a full review dashboard.

## Future Improvements

- Add a small demo dataset or sample workflow that is safe to publish.
- Add real screenshots after running on approved demo footage.
- Add an architecture diagram under `assets/`.
- Add optional progress bars for long indexing jobs.
- Add unit tests for filename sanitization and CLI argument handling.
- Add configurable model selection.
- Add a lightweight web interface for reviewing search results.

## GitHub Repo Settings

Suggested GitHub description:

```text
AI-powered local video search tool that indexes footage with CLIP embeddings and lets you find matching frames using natural-language queries.
```

Suggested topics:

```text
computer-vision
clip
embeddings
vector-search
chromadb
video-search
semantic-search
python
ai
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
