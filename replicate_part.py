import replicate.client
import streamlit as st
import replicate
import json


replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

def transcribe_audio(audio_path, lang):
    with open(audio_path, "rb") as file:
        output = replicate_client.run(
            "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
            input={
                "audio_file": file,
                "align_output": False
            }
        )

    transcript = ""

    for line in output["segments"]:
        transcript += line["text"] + " "

    return transcript
