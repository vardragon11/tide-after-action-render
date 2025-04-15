<h1>🏝️After Action Renderer</h1>

After Action Renderer is an AI-powered multimedia web app that transforms text-based wargame scenarios into cinematic simulations.
It combines scene generation, storytelling, sound design, and video editing into one cohesive platform built on Streamlit.

<h1> 🎮Core Application Flow</H1>
Text → Battle Data → Video Simulation

An AI-powered application that transforms text-based wargame scenarios into immersive rendered videos with unit data, terrain, objectives, and tactical replays.

## After Action Renderer:

- Parses the scenario into structured data (units, terrain, objectives)
- Generates images and video sequences
- Overlays music, narration, and battle events
- Compiles the final animation into a downloadable film clip

## Licenses:
#### Step 4
![Step 4](diagrams/licenses.png)

### App Functionality

Modules

Generate 
- Images Enter a prompt to create high-quality concept art using AI image generation (e.g., Stable Diffusion). Useful for environments, characters, or props.

Image-to-Image	
- Upload an image and transform part of it by drawing a mask. Fill or stylize the masked area based on your custom prompt.

Generate Dialogue 	
- Will allow you to script spoken narration or have the AI generate voiceover from context or scenario text.

Image-to-Video	
- Animate a static image using Stable Video Diffusion (SVD-xt). Great for turning illustrations into short clips with motion.

Scenario Builder
- Type in a natural language battle scene and convert it to structured battle data (scenario.json). Includes parsing for units, objectives, and terrain.

Browse Library *(coming soon)*
- Will offer a gallery of past scenarios, generated videos, prompts, and assets for reuse or remixing.

Generate Graphics
- Planned tool for auto-generating visual assets like battle maps, character cards, or scene overlays from scenario data.

🎵	Music Library	

- Browse and download background music tailored to cinematic battle scenes. Future support will include style filters and tagging.

Movie Compiler 

- Will unify video clips, narration, music, and visual overlays into a complete, timeline-edited final MP4.


📁 Project Layout
```plaintext
IPython/
├── images/
│   └── kuzco.jpg
├── manim/
│   ├── Scenario2D.py
│   ├── Scenario3D.py
│   └── scenario_utils.py
├── music/
│   ├── Action 1.mp3
│   ├── Action 2.mp3
│   ├── Ambient Tension 1.mp3
│   ├── Ambient Tension 2.mp3
│   ├── Drama 1.mp3
│   ├── Drama 2.mp3
│   ├── Lost.mp3
│   └── Win.mp3
├── pages/
│   ├── combine.py
│   ├── images-to-image.py
│   ├── manim.py
│   ├── music.py
│   ├── narrator.py
│   ├── text-to-video.py
│   ├── text.py
│   └── video.py
```

🛠️ Requirements

See the `requirements.txt` file for a detailed list of all necessary packages and version specifications.

🎬 Output

- MP4 video preview and download
- WAV narration file
- AI-generated visuals and transitions
- Timeline-driven animations

<h2>📁 Project Structure </h2>

<h3>📂 images/ </h3>
> Static assets for the Streamlit interface.

- `kuzco.jpg` – Splash or reference image for branding/UI.

---

<h2> 📂 manim/ </h2>
> Scene animation generation using [Manim](https://www.manim.community/).

- `Scenario2d.py` – 2D ground-based tactical scene animation.
- `Scenario3D.py` – 3D space-based scene builder.
- `scenario_utils.py` – Scenario parser and utility logic.

---

<h3> 📂 music/ </h3>
> Curated background music library categorized by tone.

- **Action**:  
  `Action 1.mp3`, `Action 2.mp3`
- **Ambient Tension**:  
  `Ambient Tension 1.mp3`, `Ambient Tension 2.mp3`
- **Drama**:  
  `Drama 1.mp3`, `Drama 2.mp3`
- **Victory/Loss**:  
  `Win.mp3`, `Lost.mp3`

---

<h3> 📂 pages/ *(Streamlit Application Modules)* </h3>

- `combine.py` – Combines videos, narration, and music into a cinematic sequence.
- `image-to-image.py` – Segment Anything + Inpainting-based image modification.
- `images.py` – Generates high-quality images using Stable Diffusion XL.
- `manim.py` – Scenario visualization using Manim engine.
- `music.py` – Music selection and playback.
- `narrator.py` – Generates narration from text using gTTS.
- `text-to-video.py` – Uses Zeroscope for text-to-video conversion.
- `text.py` – Scene/Dialogue writer and prompt generation.
- `video.py` – Image-to-video generation using Stable Video Diffusion.

---

<h3> 📂 sam/ </h3>
> Segment Anything model weights.

- `sam_vit_b.pth` – Pretrained model checkpoint for automatic masking.

---

<h3> 🛠️ Features </h3>

- AI-generated **images**, **videos**, and **animations**
- Text-to-speech **narration** with gTTS
- Segment Anything-powered **image masking**
- Streamlit web UI for interactive editing
- Final movie builder using **MoviePy**
- Integrated music selection from curated library

---

<h3> 📦 Powered By </h3>

- `stabilityai/stable-diffusion-xl-base-1.0`
- `stabilityai/stable-video-diffusion-img2vid-xt`
- `cerspense/zeroscope_v2_576w`
- `facebookresearch/segment-anything`
- `gTTS`, `MoviePy`, `Manim`

---

## System Requirements

This project was developed and tested using the following environment:

**Hardware:**

* **GPU:** NVIDIA Tesla T4 (Colab environment)
* **RAM:**  12 GB or more (Colab environment)
* **Disk Space:** Sufficient space for storing generated images and model data (Colab provides temporary disk space)

**Software:**

* **Operating System:** Linux-based (Colab environment)
* **Python:** 3.10.12 (Replace with your current Python version: `!python --version` in Colab)
* **Libraries:** Refer to `requirements.txt` for a comprehensive list. This includes:
    * streamlit
    * diffusers
    * Pillow
    * torch
    * localtunnel
    * streamlit
    * pydantic>=2.9
    * gtts==2.5.4
    * pydub==0.25.1
    * whisper==1.1.10
    * manim==0.17.3
    * typing
    * typing-extensions
    * libcairo2-dev 
    * libpango1.0-dev 
    * ffmpeg

**Notes:**

* This project utilizes a GPU for image generation. While you might be able to run it on a CPU, it would be significantly slower.
* Colab offers a convenient environment for running this project, providing the necessary hardware and software. However, you can also replicate the environment locally if desired.
* The `requirements.txt` file can be generated using `pip freeze > requirements.txt`. This ensures that you have the same library versions used during development.

📜 License
MIT License © 2025
Developed with ❤️ using Streamlit, Manim, and generative AI tools.


## 📘 How-To Guide (Step-by-Step Visuals)

Below are example screenshots showing how to use the tool:

### 🖼️ Step-by-Step Instructions

#### Step 1
![Step 1](how-to-guide/1.png)

#### Step 2
![Step 2](how-to-guide/2.png)

#### Step 3
![Step 3](how-to-guide/3.png)

#### Step 4
![Step 4](how-to-guide/4.png)

#### Step 5
![Step 5](how-to-guide/5.png)

#### Step 6
![Step 6](how-to-guide/6.png)

#### Step 7
![Step 7](how-to-guide/7.png)

#### Step 8
![Step 8](how-to-guide/8.png)

#### Step 9
![Step 9](how-to-guide/9.png)

#### Step 10
![Step 10](how-to-guide/10.png)
