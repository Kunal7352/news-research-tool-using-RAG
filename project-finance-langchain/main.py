import os
import time
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
# Using standard langchain.chains instead of langchain_classic if available,
# but keeping your import if you specifically need that library.
# Ideally, use: from langchain.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains import RetrievalQAWithSourcesChain

# WARNING: It is not safe to hardcode API keys in code.
# Better to use a .env file, but keeping it here as per your request.
api_key = st.secrets["OPENAI_API_KEY"]
st.title("📰 News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.9, max_tokens=500)

# 1. CHANGED: Point to the folder name used in save_local
faiss_index_path = "faiss_index"

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(documents)

    # Create embeddings + FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save FAISS index
    # This creates a FOLDER named 'faiss_index', not a single .pkl file
    vectorstore.save_local(faiss_index_path)
    main_placeholder.text("Indexing Completed! You can now ask questions.")

query = main_placeholder.text_input("Question: ")

if query:
    # 2. CHANGED: Check if the folder exists, not the old .pkl filename
    if os.path.exists(faiss_index_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.error("Vector Store not found. Please process URLs first.")
