import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

st.title("🎬 Prompt & Scene Generator with Phi-2")
st.markdown("Use this tool to generate descriptive prompts for image/video generation.")

prompt_input = st.text_area("Enter a concept, theme, or idea", "a drone flying over")

if st.button("Generate Prompt"):
    pipe = load_model()
    response = pipe(
        f"Describe in detail: {prompt_input}",
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95
    )    
    st.success("📜 Generated Prompt:")
    st.write(response[0]["generated_text"].replace(f"Describe in vivid detail: {prompt_input}", "").strip())