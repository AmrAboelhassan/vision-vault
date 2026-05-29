"""Streamlit interface for Vision Vault."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

import indexer
import searcher


APP_TITLE = "Vision Vault Studio"
APP_SUBTITLE = "Local AI-powered video search using CLIP embeddings and ChromaDB."
DEFAULT_COLLECTION_NAME = "vision_vault"
DEFAULT_DB_PATH = Path("security_db")
DEFAULT_FRAMES_PATH = Path("db_images")
DEFAULT_OUTPUT_PATH = Path("search_results")
DEFAULT_UPLOADS_PATH = Path("uploads")
SUPPORTED_VIDEO_TYPES = ("mp4", "mov", "avi", "mkv", "webm")
EXAMPLE_QUERIES = (
    "red backpack",
    "person near the door",
    "computer monitor",
    "empty hallway",
    "car in parking area",
)


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
<style>
.stApp {
    background-color: #0f0f10;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}
.vv-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 8px;
    padding: 1rem;
    background: rgba(128, 128, 128, 0.08);
    min-height: 118px;
}
.vv-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.vv-card-body {
    color: rgba(250, 250, 250, 0.72);
    font-size: 0.9rem;
    line-height: 1.35;
}
.vv-muted {
    color: rgba(250, 250, 250, 0.68);
    font-size: 0.95rem;
}
.vv-step-label {
    color: rgba(250, 250, 250, 0.62);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04rem;
    text-transform: uppercase;
}
.vv-check-row {
    align-items: center;
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.45rem;
    padding: 0.6rem 0.7rem;
}
.vv-check-label {
    font-size: 0.95rem;
    font-weight: 600;
}
.vv-status {
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
}
.vv-status-ready {
    background: rgba(46, 160, 67, 0.18);
    color: #7ee787;
}
.vv-status-missing {
    background: rgba(210, 153, 34, 0.18);
    color: #f2cc60;
}
</style>
""",
        unsafe_allow_html=True,
    )


def ensure_runtime_dependencies() -> None:
    indexer.load_runtime_dependencies()
    searcher.load_runtime_dependencies()


@st.cache_resource(show_spinner=False)
def get_clip_runtime(device_name: str) -> tuple[Any, Any, Any]:
    ensure_runtime_dependencies()
    device = indexer.resolve_device(device_name)
    model, processor = indexer.load_clip(device)
    return device, model, processor


def get_collection(collection_name: str, *, create: bool):
    ensure_runtime_dependencies()
    client = indexer.chromadb.PersistentClient(path=str(DEFAULT_DB_PATH))
    if create:
        return client.get_or_create_collection(name=collection_name)
    return client.get_collection(name=collection_name)


def sanitize_upload_filename(filename: str) -> str:
    source_path = Path(filename)
    extension = source_path.suffix.lower()
    stem = re.sub(r"[^\w\s-]", "", source_path.stem).strip().replace(" ", "_")
    return f"{stem or 'uploaded_video'}{extension}"


def save_uploaded_video(uploaded_file) -> Path:
    DEFAULT_UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_upload_filename(uploaded_file.name)
    saved_path = DEFAULT_UPLOADS_PATH / safe_name

    with saved_path.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    return saved_path


def render_step_header(step: int, title: str, helper_text: str) -> None:
    st.markdown(f'<div class="vv-step-label">Step {step}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<div class="vv-muted">{helper_text}</div>', unsafe_allow_html=True)


def render_readiness_checklist(
    *,
    video_path: Path | None,
    collection_name: str,
    sample_every: float,
) -> None:
    st.markdown("**Readiness checklist**")
    checklist = (
        ("Video selected", video_path is not None and video_path.is_file(), "Needs video"),
        ("Collection name set", bool(collection_name.strip()), "Missing"),
        ("Sampling interval set", sample_every > 0, "Missing"),
    )
    for label, ready, missing_text in checklist:
        state = "Done" if ready else missing_text
        status_class = "vv-status-ready" if ready else "vv-status-missing"
        st.markdown(
            f"""
<div class="vv-check-row">
  <div class="vv-check-label">{label}</div>
  <div class="vv-status {status_class}">{state}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.write(
        "Search local video footage with natural-language queries using CLIP "
        "embeddings and vector search."
    )
    st.info(
        "Safety note: This app does not identify people, perform face recognition, "
        "or make security decisions."
    )

    card_data = (
        ("Local-first", "Runs on your machine. No cloud upload."),
        ("Semantic search", "Find frames using natural-language queries."),
        ("Safe prototype", "No face recognition. No identity detection."),
    )
    columns = st.columns(3)
    for column, (title, body) in zip(columns, card_data):
        with column:
            st.markdown(
                f"""
<div class="vv-card">
  <div class="vv-card-title">{title}</div>
  <div class="vv-card-body">{body}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("How It Works")
        st.write(
            "Vision Vault Studio keeps the workflow local: choose a video, index "
            "sampled frames, then search the index with a text description."
        )

        st.subheader("Architecture")
        st.code(
            "Video -> Sample Frames -> CLIP Image Embeddings -> ChromaDB -> "
            "Text Query -> CLIP Text Embedding -> Similarity Search -> Matching Frames",
            language="text",
        )

        st.subheader("Limitations")
        st.markdown(
            """
- Semantic matches are not guaranteed detections.
- Accuracy depends on CLIP.
- Not real-time yet.
- No face recognition.
- No identity detection.
- No security decisions.
"""
        )


def render_index_section() -> None:
    with st.container(border=True):
        render_step_header(
            1,
            "Add video",
            "Upload a video for an easy demo, or point to a local file for larger footage.",
        )

        input_mode = st.radio(
            "Choose how to provide your video",
            options=("Upload Video", "Use Local Path"),
            horizontal=True,
        )

        selected_video_path: Path | None = None
        extension: str | None = None
        video_extension_supported = True

        if input_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=list(SUPPORTED_VIDEO_TYPES),
                help="Supported formats: mp4, mov, avi, mkv, webm.",
            )
            st.caption("Large uploads may take a little time to save before indexing.")
            st.caption("Uploaded videos are saved locally in `uploads/` and ignored by Git.")

            if uploaded_file is None:
                st.info("No video selected yet.")
            else:
                extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
                if extension not in SUPPORTED_VIDEO_TYPES:
                    video_extension_supported = False
                    st.error("Unsupported video format. Please upload mp4, mov, avi, mkv, or webm.")
                else:
                    upload_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 'unknown')}"
                    cached_upload_path = Path(st.session_state.get("uploaded_video_path", ""))
                    if (
                        st.session_state.get("uploaded_video_signature") == upload_signature
                        and cached_upload_path.is_file()
                    ):
                        selected_video_path = cached_upload_path
                    else:
                        selected_video_path = save_uploaded_video(uploaded_file)
                        st.session_state["uploaded_video_signature"] = upload_signature
                        st.session_state["uploaded_video_path"] = str(selected_video_path)
                    st.success("Video uploaded and saved locally.")
                    st.text_input("Selected file", value=uploaded_file.name, disabled=True)
                    st.text_input("Saved local path", value=str(selected_video_path), disabled=True)
        else:
            video_path_text = st.text_input(
                "Use Local Path",
                placeholder=r"C:\path\to\video.mp4",
                help="Best for large videos that are already on your machine.",
            )
            st.caption("Use this option for large files or videos already stored locally.")
            if not video_path_text.strip():
                st.info("No video selected yet.")
            else:
                selected_video_path = Path(video_path_text.strip().strip('"'))
                extension = selected_video_path.suffix.lower().lstrip(".")
                st.text_input("Selected video", value=str(selected_video_path), disabled=True)
                if extension not in SUPPORTED_VIDEO_TYPES:
                    video_extension_supported = False
                    st.error("Unsupported video format. Please use mp4, mov, avi, mkv, or webm.")

    with st.container(border=True):
        render_step_header(
            2,
            "Index footage",
            "Sample frames from the selected video and store searchable embeddings locally.",
        )

        settings_left, settings_right = st.columns(2)
        with settings_left:
            sample_every = st.number_input(
                "Frame sampling interval",
                min_value=0.1,
                max_value=60.0,
                value=2.0,
                step=0.5,
                help="Seconds between indexed frames.",
            )
            collection_name = st.text_input("Collection/index name", value=DEFAULT_COLLECTION_NAME)
        with settings_right:
            device_name = st.selectbox("Device", options=("auto", "cuda", "cpu"), index=0)
            render_readiness_checklist(
                video_path=selected_video_path,
                collection_name=collection_name,
                sample_every=sample_every,
            )

        submitted = st.button("Index Video", type="primary")

    if not submitted:
        return

    video_path = selected_video_path
    collection_name = collection_name.strip() or DEFAULT_COLLECTION_NAME

    if video_path is None:
        st.error("Please upload a video or provide a valid local video path first.")
        return
    if not video_extension_supported:
        st.error("Unsupported video format. Please use mp4, mov, avi, mkv, or webm.")
        return
    if not video_path.is_file():
        st.error("Please upload a video or provide a valid local video path first.")
        return

    DEFAULT_FRAMES_PATH.mkdir(parents=True, exist_ok=True)
    DEFAULT_DB_PATH.mkdir(parents=True, exist_ok=True)

    status = st.status("Preparing indexer...", expanded=True)
    try:
        with st.spinner("Loading CLIP model..."):
            device, model, processor = get_clip_runtime(device_name)

        collection = get_collection(collection_name, create=True)
        status.write(f"Collection: {collection_name}")
        status.write(f"Video: {video_path}")
        status.write(f"Sampling every {sample_every:g} seconds")

        progress_bar = st.progress(0, text="Waiting to index frames...")

        def update_progress(progress: float, frame_number: int, indexed_count: int) -> None:
            progress_bar.progress(
                min(progress, 1.0),
                text=f"Indexed {indexed_count:,} frames. Current frame: {frame_number:,}",
            )

        with st.spinner("Indexing video. This can take a while for long files."):
            indexed_count = indexer.index_video(
                video_path,
                collection=collection,
                model=model,
                processor=processor,
                device=device,
                frames_path=DEFAULT_FRAMES_PATH,
                sample_every=sample_every,
                progress_callback=update_progress,
            )

        progress_bar.progress(1.0, text="Indexing complete.")
        status.update(label="Indexing complete", state="complete")
        if indexed_count:
            st.success(f"Indexed {indexed_count:,} frames into '{collection_name}'.")
            st.session_state["vision_vault_collection"] = collection_name
        else:
            st.warning("No frames were indexed. Check that the video can be opened.")
    except Exception as exc:
        status.update(label="Indexing failed", state="error")
        st.error(f"Indexing failed: {exc}")


def render_result_card(match: dict[str, Any]) -> None:
    image_path = match.get("result_path")
    if image_path and Path(image_path).is_file():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info("Frame preview unavailable.")

    st.metric("Score", f"{match.get('score', 0):.2f}")
    st.caption(f"Timestamp: {match.get('timestamp', 'unknown')}")
    st.caption(f"Source video: {match.get('source', 'unknown')}")
    st.caption(f"Saved result path: {image_path or 'unknown'}")


def render_results_section() -> None:
    with st.container(border=True):
        render_step_header(
            4,
            "Review matching frames",
            "Matching frames are exported locally and shown here after a search.",
        )

        search_ran = st.session_state.get("search_ran", False)
        matches = st.session_state.get("last_matches", [])

        if not search_ran:
            st.info("Search results will appear here after you run a query.")
            return

        if not matches:
            st.info(
                "No matching frames found. Try a broader query or reduce the "
                "similarity threshold if available."
            )
            return

        st.success(f"Found {len(matches)} matching frames.")
        st.caption("Results are saved locally in `search_results/` and ignored by Git.")

        columns = st.columns(3)
        for index, match in enumerate(matches):
            with columns[index % len(columns)]:
                with st.container(border=True):
                    render_result_card(match)


def render_search_section() -> None:
    with st.container(border=True):
        render_step_header(
            3,
            "Search with natural language",
            "Search your indexed footage using natural-language descriptions.",
        )

        st.caption("Example queries")
        example_columns = st.columns(len(EXAMPLE_QUERIES))
        for column, example_query in zip(example_columns, EXAMPLE_QUERIES):
            with column:
                if st.button(example_query, key=f"example-{searcher.sanitize_filename(example_query)}"):
                    st.session_state["search_query"] = example_query

        default_collection = st.session_state.get("vision_vault_collection", DEFAULT_COLLECTION_NAME)
        with st.form("search-form"):
            query = st.text_input(
                "Natural-language query",
                placeholder="red backpack",
                key="search_query",
            )
            top_k = st.slider("Top K", min_value=1, max_value=200, value=50, step=1)
            collection_name = st.text_input("Collection/index name", value=default_collection)
            device_name = st.selectbox("Search device", options=("auto", "cuda", "cpu"), index=0)
            submitted = st.form_submit_button("Search", type="primary")

    if not submitted:
        return

    query = query.strip()
    collection_name = collection_name.strip() or DEFAULT_COLLECTION_NAME

    if not query:
        st.error("Enter a natural-language query before searching.")
        return

    status = st.status("Preparing search...", expanded=True)
    try:
        with st.spinner("Loading CLIP model..."):
            device, model, processor = get_clip_runtime(device_name)

        collection = get_collection(collection_name, create=False)
        if collection.count() == 0:
            status.update(label="Search stopped", state="error")
            st.error(f"Collection '{collection_name}' is empty. Index a video first.")
            return

        status.write(f"Collection: {collection_name}")
        status.write(f"Query: {query}")
        status.write(f"Inspecting top {top_k} candidates")

        with st.spinner("Searching indexed frames..."):
            matches = searcher.search_matches(
                query,
                collection=collection,
                model=model,
                processor=processor,
                device=device,
                output_path=DEFAULT_OUTPUT_PATH,
                top_k=top_k,
                threshold=0.80,
            )

        status.update(label="Search complete", state="complete")
        st.session_state["search_ran"] = True
        st.session_state["last_matches"] = matches
        st.session_state["last_query"] = query
    except Exception as exc:
        status.update(label="Search failed", state="error")
        st.error(
            f"Search failed: {exc}. Make sure the collection exists and has been indexed."
        )


def main() -> None:
    configure_page()
    render_header()
    render_sidebar()

    render_index_section()
    st.divider()
    render_search_section()
    st.divider()
    render_results_section()


if __name__ == "__main__":
    main()
