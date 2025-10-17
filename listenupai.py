# # # # import streamlit as st
# # # # from transformers import pipeline
# # # # from gtts import gTTS
# # # # import uuid
# # # # import os

# # # # st.set_page_config(page_title="🎧 ListenUpAI Lite", layout="centered")
# # # # st.title("🎧 ListenUpAI Lite — Expressive Audiobook Generator")

# # # # # ---------------- Model ----------------
# # # # @st.cache_resource
# # # # def load_model():
# # # #     return pipeline("text-generation", model="gpt2")  # small & free

# # # # generator = load_model()

# # # # # ---------------- User Input ----------------
# # # # text_input = st.text_area("Paste or type your text here", height=200)
# # # # tone = st.selectbox("Choose Tone", ["Neutral", "Inspiring", "Suspenseful"])

# # # # # ---------------- Generate ----------------
# # # # if st.button("🎙️ Generate Audiobook"):
# # # #     if text_input.strip():
# # # #         with st.spinner("📝 Generating text..."):
# # # #             result = generator(text_input, max_length=200)
# # # #             rewritten_text = result[0]["generated_text"]

# # # #         # Display side-by-side
# # # #         st.subheader("🔹 Original vs Rewritten Text")
# # # #         col1, col2 = st.columns(2)
# # # #         with col1:
# # # #             st.markdown("**Original:**")
# # # #             st.write(text_input)
# # # #         with col2:
# # # #             st.markdown(f"**{tone} Tone Rewrite:**")
# # # #             st.write(rewritten_text)

# # # #         # ---------------- Audio ----------------
# # # #         st.subheader("🎧 Playback & Download")
# # # #         audio_file = f"listenupai_{uuid.uuid4().hex}.mp3"
# # # #         tts = gTTS(text=rewritten_text, lang="en")
# # # #         tts.save(audio_file)
# # # #         st.audio(audio_file, format="audio/mp3")
# # # #         st.download_button("⬇️ Download Audio", open(audio_file, "rb"), file_name="listenupai_audio.mp3")
# # # #         os.remove(audio_file)
# # # #     else:
# # # #         st.warning("⚠️ Please enter some t



# # # import streamlit as st
# # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # # import torch
# # # from gtts import gTTS
# # # import os

# # # # Load IBM Granite model (small + free)
# # # MODEL_NAME = "ibm/granite-3b-code-instruct"
# # # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# # # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# # # # Streamlit App
# # # st.title("🎧 ListenUpAI — Text to Expressive Audiobook Generator")

# # # st.markdown("Convert your written text into **expressive audiobooks** with tone adaptation and natural narration!")

# # # # Tone selection
# # # tone = st.selectbox("Choose Tone", ["Neutral", "Inspiring", "Suspenseful"])

# # # # Text input
# # # text_input = st.text_area("Paste or type your text here", height=200)

# # # # Function to rewrite text with tone
# # # def rewrite_text(text, tone):
# # #     prompt = f"Rewrite the following text in a {tone} tone, keeping the meaning same:\n\n{text}"
# # #     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
# # #     with torch.no_grad():
# # #         outputs = model.generate(**inputs, max_new_tokens=400)
# # #     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # # # Generate rewritten + audio
# # # if st.button("🎙️ Generate Audiobook"):
# # #     if text_input.strip():
# # #         rewritten_text = rewrite_text(text_input, tone)
# # #         st.subheader("🔹 Tone-Adaptive Rewrite")
# # #         st.write(rewritten_text)

# # #         # Generate audio using gTTS
# # #         tts = gTTS(text=rewritten_text)
# # #         audio_path = "listenupai_output.mp3"
# # #         tts.save(audio_path)

# # #         # Play audio in app
# # #         st.audio(audio_path)
# # #         st.download_button("⬇️ Download Audio", open(audio_path, "rb"), file_name="listenupai_output.mp3")

# # #     else:
# # #         st.warning("⚠️ Please enter some text first!")






# # import streamlit as st
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # from gtts import gTTS
# # import os
# # import uuid
# # import re

# # # ------------------ Load IBM Granite-like model ------------------
# # # Using HuggingFace FLAN-T5 small as a free substitute
# # @st.cache_resource
# # def load_model():
# #     model_name = "google/flan-t5-small"
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# #     return tokenizer, model

# # tokenizer, model = load_model()

# # # ------------------ Streamlit App ------------------
# # st.set_page_config(page_title="🎧 ListenUpAI / EchoVerse", layout="wide")
# # st.title("🎧 ListenUpAI — Expressive Audiobook Generator")
# # st.markdown(
# #     "Transform your text into **engaging audiobooks** with customizable tone and voice."
# # )

# # # Sidebar: Settings
# # st.sidebar.header("Settings")
# # tones = ["Neutral", "Inspiring", "Suspenseful"]
# # selected_tone = st.sidebar.selectbox("Select Tone", tones)

# # voices_tld = {"US": "com", "UK": "co.uk", "India": "co.in"}
# # selected_voice = st.sidebar.selectbox("Select Voice Accent", list(voices_tld.keys()))

# # speed = st.sidebar.slider("Playback Speed", 0.5, 2.0, 1.0, 0.1)

# # # ------------------ Text Input ------------------
# # text_input = st.text_area("Paste or type your text here:", height=200)

# # # ------------------ Helper Functions ------------------
# # def expand_and_rewrite(text, tone):
# #     """Rewrite and expand text paragraph by paragraph."""
# #     paragraphs = text.split("\n\n")
# #     rewritten_paragraphs = []

# #     for p in paragraphs:
# #         if not p.strip():
# #             continue

# #         prompt = f"""
# # Rewrite this paragraph in a {tone} tone.
# # - Expand it with examples, details, storytelling.
# # - Keep the meaning the same.
# # - Make it engaging for audiobook narration.

# # Paragraph:
# # {p}
# # """
# #         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=700,
# #             do_sample=True,
# #             top_p=0.95,
# #             temperature=0.85
# #         )
# #         rewritten_paragraphs.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
# #     return "\n\n".join(rewritten_paragraphs)

# # def split_sentences(text):
# #     return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# # def text_to_speech(text, filename, tld="com"):
# #     tts = gTTS(text=text, lang="en", tld=tld)
# #     tts.save(filename)

# # # ------------------ Generate Button ------------------
# # if st.button("🎙️ Generate Audiobook"):
# #     if not text_input.strip():
# #         st.warning("⚠️ Please enter some text!")
# #     else:
# #         with st.spinner("✍️ Rewriting and expanding text..."):
# #             rewritten_text = expand_and_rewrite(text_input, selected_tone)

# #         # ------------------ Side-by-Side Display ------------------
# #         st.subheader("🔹 Original vs Expanded Rewrite")
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.markdown("**Original Text:**")
# #             st.markdown(
# #                 f"<div style='background:#f0f8ff; color:#000; padding:15px; border-radius:10px'>{text_input}</div>",
# #                 unsafe_allow_html=True,
# #             )
# #         with col2:
# #             st.markdown(f"**{selected_tone} Tone Rewrite:**")
# #             st.markdown(
# #                 f"<div style='background:#ffe4e1; color:#000; padding:15px; border-radius:10px'>{rewritten_text}</div>",
# #                 unsafe_allow_html=True,
# #             )

# #         # ------------------ Generate Audio ------------------
# #         st.subheader("🎧 Audio Playback & Download")
# #         audio_file = f"listenupai_{uuid.uuid4().hex}.mp3"
# #         text_to_speech(rewritten_text, audio_file, voices_tld[selected_voice])
# #         st.audio(audio_file, format="audio/mp3")
# #         st.download_button(
# #             "⬇️ Download Audio", open(audio_file, "rb"), file_name="listenupai_audio.mp3"
# #         )
# #         os.remove(audio_file)

# #         # ------------------ Text Analytics ------------------
# #         sentences = split_sentences(rewritten_text)
# #         word_count = len(rewritten_text.split())
# #         sentence_count = len(sentences)
# #         est_duration_sec = max(word_count / 2, 1)  # ~2 words/sec reading speed
# #         st.subheader("📊 Text Analytics")
# #         st.markdown(
# #             f"**Words:** {word_count} | **Sentences:** {sentence_count} | **Estimated Duration:** {est_duration_sec:.1f} sec"
# #         )
# # def expand_and_rewrite(text, tone):
# #     """Rewrite and expand text paragraph by paragraph for better output."""
# #     paragraphs = text.split("\n")
# #     rewritten_paragraphs = []

# #     for p in paragraphs:
# #         if not p.strip():
# #             continue

# #         # Use concise prompt so model understands clearly
# #         prompt = f"Rewrite the following text in a {tone} tone and make it engaging for audiobook narration:\n{p}"

# #         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=200,  # small but sufficient for small inputs
# #             do_sample=True,
# #             top_p=0.95,
# #             temperature=0.9
# #         )

# #         rewritten_paragraphs.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

# #     return "\n".join(rewritten_paragraphs)
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from gtts import gTTS
# import os
# import uuid
# import re

# # ------------------ Load FLAN-T5 Large (8-bit) ------------------
# @st.cache_resource
# def load_model():
#     model_name = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         load_in_8bit=True
#     )
#     return tokenizer, model

# tokenizer, model = load_model()

# # ------------------ Streamlit App ------------------
# st.set_page_config(page_title="🎧 ListenUpAI / EchoVerse", layout="wide")
# st.title("🎧 ListenUpAI — Expanded Audiobook Generator")
# st.markdown(
#     "Transform your text into **rich, audiobook-ready narration** with selectable tone and voice."
# )

# # ------------------ Sidebar Settings ------------------
# st.sidebar.header("Settings")
# tones = ["Neutral", "Inspiring", "Suspenseful"]
# selected_tone = st.sidebar.selectbox("Select Tone", tones)

# voices_tld = {"US": "com", "UK": "co.uk", "India": "co.in"}
# selected_voice = st.sidebar.selectbox("Select Voice Accent", list(voices_tld.keys()))

# speed = st.sidebar.slider("Playback Speed", 0.5, 2.0, 1.0, 0.1)

# # ------------------ Text Input ------------------
# text_input = st.text_area("Paste or type your text here:", height=200)

# # ------------------ Helper Functions ------------------
# def expand_and_rewrite(text, tone):
#     """Rewrite and expand text paragraph by paragraph for audiobook-style output."""
#     paragraphs = text.split("\n")
#     rewritten_paragraphs = []

#     for p in paragraphs:
#         if not p.strip():
#             continue

#         prompt = f"""
# Rewrite the following text in a {tone} tone for audiobook narration.
# - Expand with examples, details, or storytelling.
# - Keep original meaning intact.
# - Make it engaging and immersive.

# Text: {p}
# """

#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=500,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.9
#         )

#         rewritten_paragraphs.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

#     return "\n\n".join(rewritten_paragraphs)

# def split_sentences(text):
#     return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# def text_to_speech(text, filename, tld="com"):
#     tts = gTTS(text=text, lang="en", tld=tld)
#     tts.save(filename)

# # ------------------ Generate Button ------------------
# if st.button("🎙️ Generate Audiobook"):
#     if not text_input.strip():
#         st.warning("⚠️ Please enter some text!")
#     else:
#         with st.spinner("✍️ Rewriting and expanding text..."):
#             rewritten_text = expand_and_rewrite(text_input, selected_tone)

#         # ------------------ Side-by-Side Display ------------------
#         st.subheader("🔹 Original vs Expanded Rewrite")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Original Text:**")
#             st.markdown(
#                 f"<div style='background:#f0f8ff; color:#000; padding:15px; border-radius:10px'>{text_input}</div>",
#                 unsafe_allow_html=True,
#             )
#         with col2:
#             st.markdown(f"**{selected_tone} Tone Rewrite:**")
#             st.markdown(
#                 f"<div style='background:#ffe4e1; color:#000; padding:15px; border-radius:10px'>{rewritten_text}</div>",
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from gtts import gTTS
# import os
# import uuid
# import re

# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from gtts import gTTS
# import os
# import uuid
# import re

# # ----------------- Model Setup -----------------
# @st.cache_resource
# def load_model():
#     model_name = "google/flan-t5-small"  # smaller and lightweight
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return tokenizer, model

# tokenizer, model = load_model()

# # ----------------- Streamlit UI -----------------
# st.set_page_config(page_title="🎧 EchoVerse / ListenUpAI", layout="wide")
# st.title("🎧 EchoVerse — Free Audiobook Generator")

# st.markdown(
#     "Transform your text into **expanded, audiobook-ready narration** with selectable tone."
# )

# # Sidebar settings
# st.sidebar.header("Settings")
# tones = ["Neutral", "Inspiring", "Suspenseful"]
# selected_tone = st.sidebar.selectbox("Select Tone", tones)

# voice_tld = {"US": "com", "UK": "co.uk", "India": "co.in"}
# selected_voice = st.sidebar.selectbox("Select Voice Accent", list(voice_tld.keys()))

# # Text input
# text_input = st.text_area("Paste or type your text here:", height=200)

# # ----------------- Functions -----------------
# def expand_rewrite(text, tone):
#     paragraphs = text.split("\n")
#     rewritten = []

#     for p in paragraphs:
#         if not p.strip():
#             continue

#         prompt = f"""
# Rewrite the following text in a {tone} tone for audiobook narration.
# - Expand with examples, details, or storytelling.
# - Keep original meaning.
# Text: {p}
# """

#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=200,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.8
#         )
#         rewritten.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

#     return "\n\n".join(rewritten)

# def split_sentences(text):
#     return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# def text_to_speech(text, filename, tld="com"):
#     tts = gTTS(text=text, lang="en", tld=tld)
#     tts.save(filename)

# # ----------------- Generate Button -----------------
# if st.button("🎙️ Generate Audiobook"):
#     if not text_input.strip():
#         st.warning("⚠️ Please enter some text!")
#     else:
#         with st.spinner("✍️ Rewriting text..."):
#             rewritten_text = expand_rewrite(text_input, selected_tone)

#         # Side-by-side display
#         st.subheader("🔹 Original vs Expanded Text")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Original Text:**")
#             st.markdown(
#                 f"<div style='background:#f0f8ff; color:#000; padding:10px; border-radius:5px'>{text_input}</div>",
#                 unsafe_allow_html=True,
#             )
#         with col2:
#             st.markdown(f"**{selected_tone} Tone Rewrite:**")
#             st.markdown(
#                 f"<div style='background:#ffe4e1; color:#000; padding:10px; border-radius:5px'>{rewritten_text}</div>",
#                 unsafe_allow_html=True,
#             )

#         # Generate audio
#         st.subheader("🎧 Audio Playback & Download")
#         audio_file = f"listenupai_{uuid.uuid4().hex}.mp3"
#         text_to_speech(rewritten_text, audio_file, voice_tld[selected_voice])
#         st.audio(audio_file, format="audio/mp3")
#         st.download_button(
#             "⬇️ Download Audio", open(audio_file, "rb"), file_name="listenupai_audio.mp3"
#         )
#         os.remove(audio_file)

#         # Text analytics
#         sentences = split_sentences(rewritten_text)
#         word_count = len(rewritten_text.split())
#         sentence_count = len(sentences)
#         est_duration_sec = max(word_count / 2, 1)  # ~2 words/sec
#         st.subheader("📊 Text Analytics")
#         st.markdown(
#             f"**Words:** {word_count} | **Sentences:** {sentence_count} | **Estimated Duration:** {est_duration_sec:.1f} sec"
#         )













# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import requests
# from gtts import gTTS
# import tempfile
# import os

# # ------------------- APP CONFIG -------------------
# st.set_page_config(page_title="EchoVerse - AI Audiobook Creator", layout="wide")

# st.markdown(
#     "<h1 style='text-align:center; color:#00bcd4;'>🎧 EchoVerse - Generative AI Audiobook Creator</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown("<p style='text-align:center;'>Transform your text into tone-adapted audiobook narrations 🎙️</p>", unsafe_allow_html=True)

# # ------------------- SIDEBAR -------------------
# st.sidebar.title("⚙️ Settings")
# tone = st.sidebar.selectbox("Select Tone", ["Neutral", "Suspenseful", "Inspiring"])
# mode = st.sidebar.radio("Choose Model", ["Local (FLAN-T5)", "Cloud (Mistral 7B via API)"])
# voice = st.sidebar.selectbox("Voice", ["female", "male"])
# st.sidebar.markdown("---")
# st.sidebar.info("💡 Tip: For richer rewrites, use *Cloud (Mistral)* mode (requires internet).")

# # ------------------- TEXT INPUT -------------------
# st.subheader("📝 Enter or Upload Your Text")

# input_text = st.text_area("Paste your text here:", height=150, placeholder="Type or paste your story...")

# uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
# if uploaded_file:
#     input_text = uploaded_file.read().decode("utf-8")

# if not input_text.strip():
#     st.warning("Please enter or upload some text to continue.")
#     st.stop()

# # ------------------- TONE PROMPT -------------------
# prompt = f"Rewrite the following text in a {tone.lower()} tone suitable for audiobook narration. Expand it to make it engaging, expressive, and detailed:\n\n{input_text}"

# # ------------------- LOCAL MODEL FUNCTION -------------------
# @st.cache_resource
# def load_local_model():
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     return tokenizer, model

# def rewrite_local(text):
#     tokenizer, model = load_local_model()
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.8, top_p=0.9)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ------------------- CLOUD MODEL FUNCTION -------------------
# def rewrite_cloud(text):
#     HF_API_TOKEN = "YOUR_HUGGINGFACE_TOKEN"  # <--- replace with your token
#     headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
#     payload = {
#         "inputs": text,
#         "parameters": {"max_new_tokens": 600, "temperature": 0.9, "top_p": 0.9},
#     }
#     url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
#     response = requests.post(url, headers=headers, json=payload)
#     result = response.json()
#     if isinstance(result, list) and "generated_text" in result[0]:
#         return result[0]["generated_text"]
#     return str(result)

# # ------------------- PROCESS -------------------
# st.subheader("🔄 Generating Tone-Adaptive Rewrite...")
# if st.button("✨ Rewrite Text"):
#     with st.spinner("Rewriting in progress... please wait ⏳"):
#         if mode == "Local (FLAN-T5)":
#             rewritten_text = rewrite_local(prompt)
#         else:
#             rewritten_text = rewrite_cloud(prompt)
#         st.success("✅ Rewriting Complete!")

#         # ------------------- DISPLAY RESULTS -------------------
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("### 🧾 Original Text")
#             st.text_area("", input_text, height=300)
#         with col2:
#             st.markdown("### 🎭 Tone-Adapted Rewrite")
#             st.text_area("", rewritten_text, height=300)

#         # ------------------- AUDIO GENERATION -------------------
#         st.markdown("### 🔊 Generate Audio Narration")
#         tts = gTTS(rewritten_text, lang="en", slow=False, tld="com" if voice == "male" else "co.in")

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
#             tts.save(tmp_file.name)
#             st.audio(tmp_file.name, format="audio/mp3")
#             st.download_button("⬇️ Download Audio", data=open(tmp_file.name, "rb"), file_name="EchoVerse_Narration.mp3")

# st.markdown("---")
# st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using IBM Granite + Streamlit + TTS</p>", unsafe_allow_html=True)



import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

# ------------------- APP CONFIG -------------------
st.set_page_config(page_title="EchoVerse - AI Audiobook Creator", layout="wide")

st.markdown(
    "<h1 style='text-align:center; color:#00bcd4;'>🎧 EchoVerse - Generative AI Audiobook Creator</h1>",
    unsafe_allow_html=True,
)
st.markdown("<p style='text-align:center;'>Transform your text into tone-adapted audiobook narrations 🎙️</p>", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("⚙️ Settings")
tone = st.sidebar.selectbox("Select Tone", ["Neutral", "Suspenseful", "Inspiring"])
voice = st.sidebar.selectbox("Voice", ["female", "male"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Local Granite Pipeline is fully offline and CPU-friendly!")

# ------------------- TEXT INPUT -------------------
st.subheader("📝 Enter or Upload Your Text")

input_text = st.text_area("Paste your text here:", height=150, placeholder="Type or paste your story...")

uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")

if not input_text.strip():
    st.warning("Please enter or upload some text to continue.")
    st.stop()

# ------------------- TONE PROMPT -------------------
prompt = f"Rewrite the following text in a {tone.lower()} tone suitable for audiobook narration. Expand it to make it engaging, expressive, and detailed:\n\n{input_text}"

# ------------------- LOCAL GRANITE PIPELINE -------------------
@st.cache_resource
def load_granite_pipeline():
    pipe = pipeline("text-generation", model="ibm-granite/granite-3.3-2b-instruct")
    return pipe

def rewrite_with_granite_pipeline(text):
    pipe = load_granite_pipeline()
    result = pipe(text, max_new_tokens=400, do_sample=True, temperature=0.8, top_p=0.9)
    return result[0]["generated_text"]

# ------------------- PROCESS -------------------
st.subheader("🔄 Generating Tone-Adaptive Rewrite...")
if st.button("✨ Rewrite Text"):
    with st.spinner("Rewriting in progress... please wait ⏳"):
        rewritten_text = rewrite_with_granite_pipeline(prompt)
        st.success("✅ Rewriting Complete!")

        # ------------------- DISPLAY RESULTS -------------------
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧾 Original Text")
            st.text_area("", input_text, height=300)
        with col2:
            st.markdown("### 🎭 Tone-Adapted Rewrite")
            st.text_area("", rewritten_text, height=300)

        # ------------------- AUDIO GENERATION -------------------
        st.markdown("### 🔊 Generate Audio Narration")
        tld_map = {"male": "com", "female": "co.in"}
        tts = gTTS(rewritten_text, lang="en", slow=False, tld=tld_map[voice])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
            st.download_button(
                "⬇️ Download Audio",
                data=open(tmp_file.name, "rb"),
                file_name="EchoVerse_Narration.mp3"
            )

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using IBM Granite Pipeline + Streamlit + TTS</p>", unsafe_allow_html=True)
