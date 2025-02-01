import streamlit as st
import os
import tempfile
from pydub import AudioSegment
from openai import OpenAI


MAX_CHUNK_SIZE_MB = 25
MAX_CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_MB * 1024 * 1024

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def _get_chunk_duration(audio: AudioSegment, max_size_bytes=MAX_CHUNK_SIZE_BYTES):
    bytes_per_sec = len(audio.raw_data) / len(audio)
    max_chunk_duration = int(max_size_bytes / bytes_per_sec)
    return max(1, max_chunk_duration)

def _chunk_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    chunk_duration = _get_chunk_duration(audio)

    chunks = []
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunks.append(chunk)
    return chunks

def transcribe_audio(audio_path, lang):
    file_size = os.path.getsize(audio_path)
    if file_size > MAX_CHUNK_SIZE_BYTES:
        chunks = _chunk_audio(audio_path)
        transcription = _transcribe_chunks(chunks, lang)
    else:
        transcription = _transcribe_audio(audio_path, lang)

    return transcription


def _transcribe_audio(audio_path, lang):
    with open(audio_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            temperature=0.0,
            language=lang,
        )
    return transcription

def _transcribe_chunks(chunks, lang):
    transcription = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f"temp_audio_chunk{i}.wav")
            chunk.export(chunk_path, format="wav")
            transcription += _transcribe_audio(chunk_path, lang)

    return transcription
