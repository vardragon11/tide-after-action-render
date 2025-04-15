import streamlit as st

from diffusers import DiffusionPipeline
import torch

# UI setup
st.sidebar.header("📷 Image Generation")
st.title("🏝️  AI Image Generation")

# Load the pipeline once
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    return pipe

pipe = load_pipeline()

# Prompt input
prompt_input = st.text_area("Enter your image prompt:", 
                            "a photorealistic satellite orbiting Earth in outer space, detailed metallic textures, solar panels reflecting sunlight, Earth visible in the background with clouds and curvature, realistic lighting and shadows, high detail, sharp focus, 8k resolution, NASA-style, shot from a floating space camera, deep space background with stars")

# Generate button
if st.button("🚀 Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt=prompt_input).images[0]
        st.image(image, caption=prompt_input, use_column_width=True)


