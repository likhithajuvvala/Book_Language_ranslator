import streamlit as st
from transformers import pipeline
from langdetect import detect

st.set_page_config(page_title="Book Language Translator", layout="centered")

st.title("📚 Book Language Translator")
st.write("Translate book text using Machine Learning (Transformer Models)")

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Tamil": "ta"
}

@st.cache_resource
def load_translator(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    return pipeline("translation", model=model_name)

text = st.text_area("Enter book text or paragraph", height=200)
target_language = st.selectbox("Select target language", list(LANGUAGES.keys()))

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text to translate.")
    else:
        try:
            src_lang = detect(text)
            tgt_lang = LANGUAGES[target_language]

            with st.spinner("Translating..."):

                # Case 1: Direct translation exists
                if src_lang == "en" or tgt_lang == "en":
                    translator = load_translator(src_lang, tgt_lang)
                    output = translator(text, max_length=512)[0]["translation_text"]

                # Case 2: Pivot through English
                else:
                    to_english = load_translator(src_lang, "en")
                    english_text = to_english(text, max_length=512)[0]["translation_text"]

                    from_english = load_translator("en", tgt_lang)
                    output = from_english(english_text, max_length=512)[0]["translation_text"]

            st.success("Translation completed!")
            st.text_area("Translated Text", output, height=200)

        except Exception as e:
            st.error("Translation failed due to unsupported language model.")
