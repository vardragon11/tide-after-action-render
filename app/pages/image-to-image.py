import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2

# Import Segment Anything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

st.title("🪄 Image Inpainting with Segment Anything + Stable Diffusion")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe

pipe = load_pipeline()

# Load SAM model
@st.cache_resource
def load_sam():
    sam_checkpoint = "/content/sam/sam_vit_b.pth"  # You must have this checkpoint in working directory
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return SamAutomaticMaskGenerator(sam.to("cuda"))

mask_generator = load_sam()

# Upload image
st.header("1. Upload Image")
uploaded_image = st.file_uploader("Choose an image (512x512 recommended)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB").resize((512, 512))
    st.image(image, caption="Original", use_column_width=True)

    st.header("2. Automatically Generate Mask using SAM")

    image_np = np.array(image)
    masks = mask_generator.generate(image_np)

    if len(masks) == 0:
        st.warning("No mask was found. Try a different image.")
    else:
        # Use the largest mask (most foreground-like)
        best_mask = max(masks, key=lambda x: x['area'])['segmentation']
        mask_img = Image.fromarray((best_mask * 255).astype(np.uint8)).convert("L")
        st.image(mask_img, caption="🩹 Auto-Generated Mask", use_column_width=True)

        st.header("3. 🎨 Inpainting Prompt")
        prompt = st.text_input("Describe what you want in the masked area:", "a vintage car parked on the road")

        if st.button("✨ Inpaint"):
            with st.spinner("Running inpainting..."):
                result_image = pipe(prompt=prompt, image=image, mask_image=mask_img).images[0]
                st.image(result_image, caption="🖼️ Inpainted Result", use_column_width=True)

                # Download
                buffer = BytesIO()
                result_image.save(buffer, format="PNG")
                st.download_button("Download Image", buffer.getvalue(), file_name="inpainted.png", mime="image/png")
else:
    st.info("📤 Upload an image to begin.")
