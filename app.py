import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

st.set_page_config(page_title="Book Language Translator", layout="centered")

st.title("📘 Book Language Translator")
st.write("Multilingual translation using Transformer-based Machine Learning")

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Tamil": "ta"
}

@st.cache_resource
def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter book text", height=200)

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source Language", LANGUAGES.keys())
with col2:
    tgt_lang = st.selectbox("Target Language", LANGUAGES.keys())

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter text to translate.")
    elif src_lang == tgt_lang:
        st.warning("Source and target languages must be different.")
    else:
        try:
            tokenizer.src_lang = LANGUAGES[src_lang]
            encoded = tokenizer(text, return_tensors="pt")

            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(LANGUAGES[tgt_lang]),
                max_length=512
            )

            translated_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            st.success("Translation successful")
            st.text_area("Translated Text", translated_text, height=200)

        except Exception as e:
            st.error("Translation failed. Please try shorter text.")
