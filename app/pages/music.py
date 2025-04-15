import streamlit as st
import os

st.title("🎵 Music Library for Your Movie")

st.markdown("""
Browse through a collection of background music tracks you can use in your video projects.
Select, preview, and download audio files directly.
""")

# Path to your local music folder
MUSIC_FOLDER = "/content/music"

# Make sure the folder exists
os.makedirs(MUSIC_FOLDER, exist_ok=True)

# Sample files (Add your own MP3/WAV files here)
sample_tracks = {
    "Epic Cinematic Theme 1": "Action 1.mp3",
    "Epic Cinematic Theme 2": "Action 2.mp3",
    "Drama Theme 1": "Drama 1.mp3",
    "Drama Theme 2": "Drama 2.mp3",
    "Ambient Tension 1": "Ambient Tension 1.mp3",
    "Ambient Tension 2": "Ambient Tension 2.mp3",
    "End Win": "Win.mp3",
    "End Lost": "Lost.mp3"
}

for title, filename in sample_tracks.items():
    audio_path = os.path.join(MUSIC_FOLDER, filename)

    if os.path.exists(audio_path):
        st.subheader(f"🎼 {title}")
        st.audio(audio_path)

        with open(audio_path, "rb") as audio_file:
            st.download_button(
                label=f"⬇️ Download {title}",
                data=audio_file.read(),
                file_name=filename,
                mime="audio/mpeg"
            )
    else:
        st.warning(f"File `{filename}` not found in `{MUSIC_FOLDER}`. Upload it manually or place it in that folder.")
