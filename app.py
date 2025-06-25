import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="📄 PDF Chatbot", page_icon="🤖")
st.title("📄 Ask Questions from Your PDF")

# Sidebar to upload PDF
pdf_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    st.sidebar.success("PDF uploaded successfully!")

    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load_and_split()

    # Create Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()

    # Load lightweight QA model
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False
    )

    st.success("Chatbot is ready! Ask your questions below.")

    # Chat interface
    user_question = st.text_input("Enter your question:")
    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": user_question})
            st.write("**Answer:**", result["result"])
else:
    st.info("Upload a PDF to get started.")

