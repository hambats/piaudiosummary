import os
from transformers import pipeline
import whisper

# Pre-load models to ensure they are cached locally

def ensure_models():
    """Download required models on first run."""
    # Whisper model
    whisper.load_model("small")
    # Huggingface summarization model
    pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

