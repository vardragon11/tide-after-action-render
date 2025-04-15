import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

st.title("🗣️ Generate Dialogue Narration")
st.markdown("Enter text below and convert it into spoken dialogue using Google Text-to-Speech.")

# Text input
narated_text = st.text_area("📝 Enter text to narrate:", height=200)

if st.button("🔊 Generate Voice"):
    if narated_text.strip() == "":
        st.warning("Please enter some text to generate speech.")
    else:
        with st.spinner("Generating audio..."):
            # Save to temporary MP3 file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
                tts = gTTS(narated_text)
                tts.save(mp3_file.name)
                mp3_path = mp3_file.name

            # Convert to WAV using pydub
            wav_path = mp3_path.replace(".mp3", ".wav")
            AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")

        st.success("✅ Audio generated!")
        st.audio(wav_path)

        # Download button
        with open(wav_path, "rb") as f:
            st.download_button("⬇️ Download Narration (WAV)", f, file_name="dialogue.wav", mime="audio/wav")
