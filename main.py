import streamlit as st
import tempfile
import os

import openai_part
import replicate_part


st.title("Transcriber")
st.sidebar.header("Upload your audio file")
uploaded_file = st.sidebar.file_uploader(label="", type=["wav", "mp3", "m4a"])

st.sidebar.header("Transcription settings")
type_of_transcription = st.sidebar.selectbox("Select type of transcription", ["OpenAI", "Replicate"], index=1)
lang = st.sidebar.selectbox("Select language", ["en", "uk", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "tr", "zh"], index=1)


if uploaded_file is not None:
    st.audio(uploaded_file.read(), format="audio/wav")

    if st.button("Transcribe", use_container_width=True):
        transcription = ""
        with st.spinner("Transcribing..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if type_of_transcription == "OpenAI":
                    transcription = openai_part.transcribe_audio(temp_audio_path, lang)
                else:
                    transcription = replicate_part.transcribe_audio(temp_audio_path, lang)

        st.success("Transcription complete!")
        st.text_area("Text form your audio", transcription, height=500)
