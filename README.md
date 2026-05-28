# AI Video Search

A local computer-vision search tool that indexes video frames with OpenAI CLIP embeddings and stores them in ChromaDB. After indexing, you can search the footage with natural-language prompts such as `red shirt`, `computer monitor`, or `person near the door`, then export matching frames for review.

## Features

- Indexes one or more video files into a local vector database.
- Uses CLIP image/text embeddings for semantic search.
- Exports high-confidence matching frames into organized result folders.
- Supports GPU acceleration with CUDA when available.
- Keeps generated frames, databases, videos, and experiment files out of Git by default.

## Project Files

| File | Purpose |
| --- | --- |
| `indexer.py` | Samples video frames and stores their CLIP embeddings in ChromaDB. |
| `searcher.py` | Searches the indexed frames using a text query and exports matches. |
| `requirements.txt` | Python dependencies for the project. |

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

The default dependency file installs PyTorch CUDA 12.1 wheels. If you do not have a CUDA-compatible GPU, install the CPU PyTorch build from the official PyTorch instructions, then install the remaining dependencies.

## Usage

Index videos:

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

Show the first match in an OpenCV preview window:

```powershell
python searcher.py "red shirt" --preview
```

Useful options:

```powershell
python indexer.py --help
python searcher.py --help
```

## Generated Data

These folders are created locally and intentionally ignored by Git:

- `security_db/` - ChromaDB vector database.
- `db_images/` - sampled frames from indexed videos.
- `search_results/` - exported search matches.
- `rtx_search_results/` - older exported result folder.

Video files are also ignored because they are usually large and may contain private footage. Add small demo media only if you have the right to publish it.

## Recommended GitHub Upload

From this folder:

```powershell
git init
git add README.md requirements.txt indexer.py searcher.py .gitignore .gitattributes
git commit -m "Prepare AI video search project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub account and repository name.
