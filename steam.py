import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr
from gtts import gTTS
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------- Load RAG -------------------
@st.cache_resource
def load_rag():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag_index", embedding_model, allow_dangerous_deserialization=True)

    model_dir = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto")

    text_gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        num_beams=4,
        repetition_penalty=1.8,
        no_repeat_ngram_size=3
    )
    llm = HuggingFacePipeline(pipeline=text_gen)

    return db, llm

db, llm = load_rag()
st.set_page_config(page_title="🎤 Tutor RAG Voice Assistant", layout="wide")
st.title("🎓 Tutor RAG Voice Assistant")

# ------------------- UI -------------------
level = st.selectbox("Choose Level:", ["Beginner", "Intermediate", "Advanced"])
st.info("🎤 Record your question below and then stop to get the answer.")

# ------------------- Audio Recorder -------------------
audio = audiorecorder("🎙 Start Recording", "⏹ Stop Recording")

if len(audio) > 0:
    # Export AudioSegment to WAV bytes
    wav_path = tempfile.mktemp(suffix=".wav")
    audio.export(wav_path, format="wav")

    # Play back in UI
    st.audio(wav_path, format="audio/wav")

    # ------------------- Speech Recognition -------------------
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            question = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            question = ""
            st.error("❌ Could not understand your speech.")
        except sr.RequestError:
            question = ""
            st.error("❌ Speech recognition service unavailable.")

    if question:
        # Capitalize first letter of each word
        question = question.title()
        st.markdown(f"🗣 *You said:* {question}")

        # ------------------- RAG -------------------
        retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"level": level}})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        st.subheader("📌 Answer")
        st.write(answer)

        # ------------------- Text-to-Speech -------------------
        tts = gTTS(answer)
        mp3_path = tempfile.mktemp(suffix=".mp3")
        tts.save(mp3_path)
        st.audio(mp3_path, format="audio/mp3")