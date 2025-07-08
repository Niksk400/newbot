import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")

# Cache model loading
@st.cache_resource
def load_bot(pdf_path):
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    # Embed with sentence-transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc.page_content for doc in documents]
    embeddings = model.encode(texts)
    vector_db = FAISS.from_texts(texts, embedding=model.encode)

    # QA pipeline with Transformers
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    return vector_db.as_retriever(), qa_pipeline

st.title("📄 PDF AI Chatbot (Offline, Free)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    retriever, qa_pipeline = load_bot("temp.pdf")
    st.success("✅ PDF Loaded")

    if question and st.button("Ask"):
        with st.spinner("Thinking..."):
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {question}\nAnswer:"
            result = qa_pipeline(prompt, max_length=200, do_sample=False)[0]['generated_text']
            st.write(result)
