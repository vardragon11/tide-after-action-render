import streamlit as st
from diffusers import DiffusionPipeline
import torch
import os
import json
from datetime import datetime
from PIL import Image
from catalog_image import categorize_prompt

# from catalog_image import (
#     categorize_prompt
#  )
torch.cuda.empty_cache()

# Folder paths
IMAGE_DIR = "catalog_images"
METADATA_FILE = "catalog_images/metadata.json"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load or initialize metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = []

# Title
st.sidebar.header("📷 Image Generation")
st.title("🏝️ AI Image Generation with Catalog")



# Model loader
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe


pipe = load_pipeline()

pipe.enable_model_cpu_offload() 

# Prompt input
prompt_input = st.text_area("Enter your image prompt:", 
    "a photorealistic satellite orbiting Earth, 8k, NASA-style")


# Generate
if st.button("🚀 Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt=prompt_input).images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(IMAGE_DIR, filename)

        # Gather the tags from prompt
        tags = categorize_prompt(prompt_input)
        #Save File
        image.save(filepath)


        # Save metadata
        entry = {"filename": filename, "prompt": prompt_input, "timestamp": timestamp,"tags":tags}
        metadata.append(entry)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        st.image(image, caption=prompt_input, use_column_width=True)
        st.success(f"Image saved to catalog as {filename}.")

# Gallery viewer
st.markdown("### 📚 Image Catalog")
for item in sorted(metadata, key=lambda x: x["timestamp"], reverse=True):
    img_path = os.path.join(IMAGE_DIR, item["filename"])
    if os.path.exists(img_path):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(img_path, width=120)
        with col2:
            st.markdown(f"**Prompt:** {item['prompt']}")
            st.markdown(f"🕒 *{item['timestamp']}*")
