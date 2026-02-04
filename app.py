import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Book Language Translator", layout="centered")

st.title("📘 Book Language Translator (ML)")
st.write("Translate book content using Transformer models")

# ONLY supported languages
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi"
}

@st.cache_resource
def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    return pipeline("translation", model=model_name)

text = st.text_area("Enter book text", height=200)

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("Source Language", LANGUAGES.keys())
with col2:
    target = st.selectbox("Target Language", LANGUAGES.keys())

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter text.")
    elif source == target:
        st.warning("Source and target languages must be different.")
    else:
        try:
            src = LANGUAGES[source]
            tgt = LANGUAGES[target]

            with st.spinner("Translating..."):
                translator = load_model(src, tgt)
                result = translator(text, max_length=512)[0]["translation_text"]

            st.success("Translation successful")
            st.text_area("Translated Text", result, height=200)

        except Exception:
            st.error("Model not available for this language pair.")
