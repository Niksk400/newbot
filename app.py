import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 📌 Custom wrapper to use SentenceTransformer with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# ✅ Load PDF, split, embed and return retriever + LLM
@st.cache_resource
def load_bot(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    return vector_db.as_retriever(), qa_pipeline

# 🎯 Streamlit UI
st.set_page_config(page_title="📄 PDF Q&A Chatbot", page_icon="🤖")
st.title("📄 PDF Q&A Chatbot (Offline, No OpenAI)")

uploaded_file = st.file_up

