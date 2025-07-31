import os
import queue
import sys
import sounddevice as sd
import numpy as np
import whisper
import tensorflow_hub as hub
import pandas as pd
import datetime
from scipy.io.wavfile import write
from model_manager import ensure_models

# Initialize models
ensure_models()
whisper_model = whisper.load_model("small")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def classify_and_transcribe(audio_data, samplerate):
    # Save temp WAV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"data/{timestamp}.wav"
    write(wav_path, samplerate, audio_data)

    # Classify with YAMNet
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    scores_np = scores.numpy()
    yamnet_classes = pd.read_csv(
        hub.resolve("https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv")
    )
    top_class = yamnet_classes['display_name'][scores_np.mean(axis=0).argmax()]

    # Transcribe with Whisper
    result = whisper_model.transcribe(wav_path)
    transcript = result['text'].strip()

    summary = {
        "time": timestamp,
        "class": top_class,
        "transcript": transcript
    }

    print(summary)
    with open("data/log.csv", "a") as logf:
        logf.write(f"{summary['time']},{summary['class']},{summary['transcript']}\n")

def main():
    os.makedirs("data", exist_ok=True)
    samplerate = 16000  # Hz
    blocksize = 8000

    print("[Listener] Starting...")
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while True:
            audio_chunk = q.get()
            audio_chunk = np.squeeze(audio_chunk)
            classify_and_transcribe(audio_chunk, samplerate)

if __name__ == "__main__":
    main()
