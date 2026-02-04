import streamlit as st
from transformers import pipeline
from langdetect import detect

# Page config
st.set_page_config(page_title="Book Language Translator", layout="centered")

st.title("📚 Book Language Translator")
st.write("Translate book text into multiple languages using Machine Learning")

# Supported languages
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Tamil": "ta"
}

@st.cache_resource
def load_translator(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    return pipeline("translation", model=model_name)

# Input text
text = st.text_area("Enter book text or paragraph", height=200)

target_language = st.selectbox("Select target language", list(LANGUAGES.keys()))

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        try:
            src_lang = detect(text)
            tgt_lang = LANGUAGES[target_language]

            if src_lang == tgt_lang:
                st.info("Source and target languages are the same.")
            else:
                with st.spinner("Translating..."):
                    translator = load_translator(src_lang, tgt_lang)
                    translated = translator(text, max_length=512)
                    st.success("Translation completed!")
                    st.text_area("Translated Text", translated[0]["translation_text"], height=200)

        except Exception as e:
            st.error("Translation failed. This language pair may not be supported.")
