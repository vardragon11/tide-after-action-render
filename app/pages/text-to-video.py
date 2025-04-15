# streamlit_app.py
import streamlit as st
import torch
import numpy as np
import imageio
import os
from diffusers import DiffusionPipeline
from tempfile import NamedTemporaryFile

st.title("🎥 Zeroscope AI Text to Video Generator")
st.markdown(
    """
    This app uses [Zeroscope v2 (576w)](https://huggingface.co/cerspense/zeroscope_v2_576w) to generate videos from prompts using Stable Diffusion.
    """
)

# Input prompt
prompt = st.text_area(
    "Enter your video description (max ~70 words for best results)",
    value="Photorealistic WWII scene, Dunkirk 1940. British soldiers evacuating under attack, smoke, burning vehicles, low-flying bombers, subtle motion, natural lighting, ultra-detailed, cinematic film tone.",
    height=100,
)

generate_btn = st.button("🎬 Generate Video")

if generate_btn and prompt:
    with st.spinner("Generating video... this may take 1–2 minutes ⏳"):
        # Load model
        pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
            torch_dtype=torch.float16
        ).to("cuda")

        # Config: more realism
        num_inference_steps = 50
        num_frames = 16
        height = 320
        width = 576
        fps = 6  # cinematic speed

        # Generate frames
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames
        )
        frames = result.frames[0]

        # Save to MP4
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            video_path = tmpfile.name
            imageio.mimsave(video_path, (frames * 255).astype(np.uint8), fps=fps)

        st.success("✅ Video generated!")
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button("⬇️ Download video", f, file_name="zeroscope_video.mp4", mime="video/mp4")
else:
    st.info("Enter a prompt and click **Generate Video** to begin.")
