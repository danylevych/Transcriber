import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
import tempfile
import os

MAX_CHUNK_SIZE_MB = 25
MAX_CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_MB * 1024 * 1024

lang = ""
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_chunk_duration(audio: AudioSegment, max_size_bytes=MAX_CHUNK_SIZE_BYTES):
    bytes_per_sec = len(audio.raw_data) / len(audio)
    max_chunk_duration = int(max_size_bytes / bytes_per_sec)
    return max(1, max_chunk_duration)

def chunk_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    chunk_duration = get_chunk_duration(audio)

    chunks = []
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunks.append(chunk)
    return chunks


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            temperature=0.0,
            language=lang
        )
    return transcription


def transcribe_chunks(chunks):
    transcription = ""
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"temp_audio_chunk{i}.wav")
        chunk.export(chunk_path, format="wav")
        transcription += transcribe_audio(chunk_path)
    return transcription

st.title("Transcriber")
st.sidebar.header("Upload your audio file")
uploaded_file = st.sidebar.file_uploader(label="", type=["wav", "mp3", "m4a"])

st.sidebar.header("Transcription settings")
lang = st.sidebar.selectbox("Select language", ["en", "uk", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "tr", "zh"], index=1)


if uploaded_file is not None:
    st.audio(uploaded_file.read(), format="audio/wav")

    with st.spinner("Transcribing..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            transcription = ""
            file_size = os.path.getsize(temp_audio_path)

            if file_size > MAX_CHUNK_SIZE_BYTES:
                chunks = chunk_audio(temp_audio_path)
                transcription = transcribe_chunks(chunks)
            else:
                transcription = transcribe_audio(temp_audio_path)

    st.success("Transcription complete!")
    st.text_area("Text form your audio", transcription, height=500)
