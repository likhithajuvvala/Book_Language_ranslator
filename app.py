import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, LangDetectException

st.set_page_config(page_title="Book Language Translator", layout="centered")

st.title("📘 Book Language Translator")
st.write("Automatic language detection with multilingual translation")

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Tamil": "ta"
}

SUPPORTED_CODES = set(LANGUAGES.values())

@st.cache_resource
def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter book text", height=200)
target_lang = st.selectbox("Target Language", LANGUAGES.keys())

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        try:
            detected_lang = detect(text)

            # ✅ Fallback instead of error
            if detected_lang not in SUPPORTED_CODES:
                detected_lang = "en"

            tokenizer.src_lang = detected_lang
            encoded = tokenizer(text, return_tensors="pt")

            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(LANGUAGES[target_lang]),
                max_length=512
            )

            translated_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            st.success(f"Source language used: {detected_lang}")
            st.text_area("Translated Text", translated_text, height=200)

        except LangDetectException:
            st.error("Could not detect language. Try clearer text.")
        except Exception:
            st.error("Translation failed. Try shorter text.")
