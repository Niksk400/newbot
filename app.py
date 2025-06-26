import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")

@st.cache_resource
def load_bot(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)

    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    return vector_db.as_retriever(), llm

st.title("📄 PDF AI Chatbot (Offline, Free)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    retriever, llm = load_bot("temp.pdf")
    st.success("PDF Loaded!")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {question}\nAnswer:"
            result = llm.invoke(prompt)
            st.write(result)
