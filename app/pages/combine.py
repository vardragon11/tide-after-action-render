import streamlit as st
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip
)
import tempfile
import os

st.title("🎬 Combine Multiple Videos with Dialogue and Music")
st.markdown("Upload multiple video clips, one dialogue audio, and one background music track to merge into one final video.")
st.image("/content/images/kuzco.jpg")

video_files = st.file_uploader("🎥 Upload video files (.mp4)", type=["mp4"], accept_multiple_files=True)
dialogue_audio = st.file_uploader("🗣️ Upload dialogue audio file (.mp3, .wav)", type=["mp3", "wav"])
music_audio = st.file_uploader("🎶 Upload background music file (.mp3, .wav)", type=["mp3", "wav"])

if video_files and dialogue_audio and music_audio:
    st.success(f"Uploaded {len(video_files)} video(s), 1 dialogue audio, and 1 music audio.")

    if st.button("🎬 Merge and Combine"):
        with st.spinner("Processing... Please wait..."):

            # Save and load dialogue audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_dialogue:
                temp_dialogue.write(dialogue_audio.read())
                dialogue_path = temp_dialogue.name
            dialogue_clip = AudioFileClip(dialogue_path)

            # Save and load music audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_music:
                temp_music.write(music_audio.read())
                music_path = temp_music.name
            music_clip = AudioFileClip(music_path).volumex(0.3)  # Lower volume for background

            video_clips = []
            target_resolution = None

            # First pass: get target resolution
            for i, uploaded_video in enumerate(video_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
                    temp_vid.write(uploaded_video.read())
                    video_path = temp_vid.name
                clip = VideoFileClip(video_path)

                if target_resolution is None:
                    target_resolution = clip.size  # (width, height)

                resized_clip = clip.resize(newsize=target_resolution)
                video_clips.append(resized_clip)

            # Concatenate all resized video clips
            final_video = concatenate_videoclips(video_clips)
            video_duration = final_video.duration

            # Adjust audios to match video duration
            def fit_audio(audio, duration):
                if audio.duration >= duration:
                    return audio.subclip(0, duration)
                else:
                    loops = int(duration // audio.duration) + 1
                    return concatenate_videoclips([audio] * loops).subclip(0, duration)

            dialogue_fitted = fit_audio(dialogue_clip, video_duration)
            music_fitted = fit_audio(music_clip, video_duration)

            # Mix both audio tracks
            mixed_audio = CompositeAudioClip([dialogue_fitted, music_fitted])

            # Set the mixed audio to video
            final_output = final_video.set_audio(mixed_audio)

            # Export final video
            output_path = os.path.join(tempfile.gettempdir(), "final_combined_video.mp4")
            final_output.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

            st.success("✅ Video created successfully!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Final Combined Video", f.read(), file_name="final_combined_video.mp4")

else:
    st.info("Please upload multiple videos and both audio files to begin.")
