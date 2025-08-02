import os
import whisper
from transformers import pipeline
import tensorflow_hub as hub

# Pre-load models to ensure they are cached locally


def ensure_models():
    """Load models if they are available locally."""
    root = os.path.expanduser(os.getenv("WHISPER_CACHE", "~/.cache/whisper"))
    try:
        whisper.load_model("small", download_root=root)
    except Exception as exc:
        print(f"Warning: could not load Whisper model: {exc}")

    try:
        pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", local_files_only=True)
    except Exception as exc:
        print(f"Warning: could not load summarization model: {exc}")

    yamnet_model = os.getenv("YAMNET_MODEL", "https://tfhub.dev/google/yamnet/1")
    try:
        hub.load(yamnet_model)
    except Exception as exc:
        print(f"Warning: could not load YAMNet model: {exc}")

