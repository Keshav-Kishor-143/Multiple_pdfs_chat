import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage 
import json
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load environment variables
load_dotenv()

# Constants
CHUNKS_PATH = "text_chunks.json"
SIMILARITY_THRESHOLD = 0.3

# Configure Streamlit page
st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

# Helper Functions
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text[:50000]

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    return text_splitter.split_text(text)

def generate_summary(text):
    """Generate a simple summary based on frequent keywords or key phrases."""
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(100)
    keywords = [word for word, freq in common_words if len(word) > 4]
    
    if "manual" in keywords and "operation" in keywords:
        return "Owner's Manual or Operational Guide"
    elif "report" in keywords or "summary" in keywords:
        return "Technical Report"
    elif "research" in keywords or "study" in keywords:
        return "Research Document"
    elif "specification" in keywords or "design" in keywords:
        return "Technical Specification Document"
    return "Document"

def store_chunks_and_create_vector_store(text_chunks):
    with open(CHUNKS_PATH, "w") as f:
        json.dump(text_chunks, f)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [chunk for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def load_existing_vector_store():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r") as f:
            text_chunks = json.load(f)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        texts = [chunk for chunk in text_chunks]
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    return None

def calculate_similarity(query_embedding, doc_embeddings):
    """Calculate cosine similarity between the query embedding and document embeddings."""
    return cosine_similarity([query_embedding], doc_embeddings).flatten()

def generate_dynamic_out_of_context_message(summary):
    return (
        f"The sources provided are technical documents, primarily {summary}. "
        "They contain information related to the document's specific content, but no relevant details were found for your query."
    )

# Main Application
def main():
    st.header("Chat with your PDFs :books:")

    with st.sidebar:
        st.subheader("Your Documents:")
        pdf_docs = st.file_uploader("Upload PDFs and click on 'Process'", accept_multiple_files=True)

        if st.button("Process New PDFs"):
            if pdf_docs:
                with st.spinner("Processing new documents..."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)
                    summary = generate_summary(raw_text)  # Generate summary
                    vector_store = store_chunks_and_create_vector_store(text_chunks)
                    st.session_state['vector_store'] = vector_store
                    st.session_state['summary'] = summary  # Store summary
                    st.success("Documents processed and added to vector store. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF document.")

        if st.button("Load Existing Documents"):
            vector_store = load_existing_vector_store()
            if vector_store:
                summary = st.session_state.get("summary", "a Document")
                st.session_state['vector_store'] = vector_store
                st.success("Loaded existing vector store. You can now ask questions.")
            else:
                st.warning("No existing documents found. Please upload and process new PDFs first.")

    # Main Chat Interface
    user_question = st.text_input("Ask a question from the uploaded documents:")

    if 'vector_store' in st.session_state and user_question:
        with st.spinner("Generating answer..."):
            vector_store = st.session_state['vector_store']
            summary = st.session_state.get("summary", "a Document")
            
            # Embed the user question for similarity-based retrieval
            openai_api_key = os.getenv("OPENAI_API_KEY")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            query_embedding = embeddings.embed_query(user_question)
            
            # Retrieve top 5 relevant documents based on the query embedding
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(user_question)
            
            # Calculate similarity scores manually
            doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
            similarity_scores = calculate_similarity(query_embedding, doc_embeddings)
            
            # Filter by threshold and sort by relevance
            relevant_docs = sorted(
                [(doc, score) for doc, score in zip(retrieved_docs, similarity_scores) if score >= SIMILARITY_THRESHOLD],
                key=lambda x: x[1],
                reverse=True
            )
            
# Assuming you have gathered top_chunks (the top relevant document chunks) based on similarity scores
            if relevant_docs:
    # Join the top relevant document chunks
             top_chunks = "\n\n".join([doc.page_content for doc, _ in relevant_docs])
    
    # Initialize the language model
             llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
    
    # Construct the message for ChatOpenAI
             user_message = HumanMessage(content=f"Answer the question '{user_question}' based on the following information:\n\n{top_chunks}")
    
    # Generate the response
             response = llm([user_message])
    
    # Extract the text content from the response
             answer = response.content
            else:
    # Fallback message if no relevant content is found
             answer = generate_dynamic_out_of_context_message(summary)

            st.write("### Answer:")
            st.write(answer)

if __name__ == '__main__':
    main()
