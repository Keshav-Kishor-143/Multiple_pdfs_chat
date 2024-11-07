import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

# Define the file path to store text chunks
CHUNKS_PATH = "text_chunks.json"

# Function to extract and truncate text from PDF files to avoid token limit issues
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    
    # Limit the text to avoid tokenization issues
    text = text[:50000]
    return text

# Function to split text into detailed chunks with overlap for rich context
def get_text_chunks_with_metadata(text, section_name=""):
    # Small chunk size with overlap for granular retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return [{"text": chunk, "metadata": {"section": section_name}} for chunk in chunks]

# Store text chunks in JSON and create FAISS vector store with metadata
def ingest_documents_with_metadata(text_chunks):
    with open(CHUNKS_PATH, "w") as f:
        json.dump(text_chunks, f)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [chunk["metadata"] for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store

# Load existing vector store if available
def get_existing_vector_store():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r") as f:
            text_chunks = json.load(f)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_texts([chunk["text"] for chunk in text_chunks], embeddings, metadatas=[chunk["metadata"] for chunk in text_chunks])
        return vector_store
    return None

# Function to create a conversation chain focused on detailed document-based answers
def get_conversation_chain(vector_store):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.5)
    
    # Low score_threshold, high k for granular retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Detailed, document-focused prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are an assistant answering questions based solely on the content of a document. "
            "Return all detailed, relevant information from the document, preserving step-by-step instructions "
            "and specific phrases. If the document contains instructions or specific procedures, include those "
            "exactly as listed.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer based only on the document content, providing comprehensive detail."
        )
    )

    # Create RetrievalQA chain with the custom prompt
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        prompt=prompt_template
    )
    
    return qa_chain

# Session state variables initialization
if "is_from_history" not in st.session_state:
    st.session_state["is_from_history"] = False
if "new_chat" not in st.session_state:
    st.session_state["new_chat"] = False

# Main app function
def main():
    st.header("Chat with multiple PDFs :books:")
    
    # Sidebar for uploading PDFs and history navigation
    with st.sidebar:
        st.subheader("Your documents here:")
        pdf_docs = st.file_uploader("Upload PDFs and click on 'Process'", accept_multiple_files=True)
        
        if "question" in st.session_state and "answer" in st.session_state:
            if st.session_state["question"] and st.session_state["answer"]:
                if st.button("🗨️ New Chat"):
                    if "selected_history" in st.session_state:
                        del st.session_state["selected_history"]
                    st.session_state["is_from_history"] = False
                    st.session_state["new_chat"] = True
                    st.session_state["question"] = ""
                    st.session_state["answer"] = ""

        if st.button("Process New PDFs"):
            if pdf_docs:
                with st.spinner("Processing new documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks_with_metadata(raw_text)
                    vector_store = ingest_documents_with_metadata(text_chunks)
                    conversation = get_conversation_chain(vector_store)
                    st.session_state['conversation'] = conversation
                    
                    if "selected_history" in st.session_state:
                        del st.session_state["selected_history"]
                    st.session_state["is_from_history"] = False

                    st.success("Documents processed and added to vector store. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF document.")

        if st.button("Load Existing Documents"):
            vector_store = get_existing_vector_store()
            if vector_store:
                conversation = get_conversation_chain(vector_store)
                st.session_state['conversation'] = conversation
                
                if "selected_history" in st.session_state:
                    del st.session_state["selected_history"]
                st.session_state["is_from_history"] = False

                st.success("Loaded existing vector store. You can now ask questions.")
            else:
                st.warning("No existing documents found. Please upload and process new PDFs first.")
        
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(st.session_state.get("history", [])):
            if st.button(f"Q{i+1}: {question[:30]}..."):
                st.session_state["selected_history"] = (question, answer)
                st.session_state["is_from_history"] = True

    if "selected_history" in st.session_state and not st.session_state.get("new_chat", False):
        selected_question, selected_answer = st.session_state["selected_history"]
        user_question = st.text_input("Ask a question from the uploaded documents:", selected_question)
    else:
        user_question = st.text_input("Ask a question from the uploaded documents:")
        st.session_state["new_chat"] = False

    if 'conversation' in st.session_state and user_question:
        if st.session_state["is_from_history"]:
            answer = selected_answer
            st.session_state["is_from_history"] = False
        else:
            with st.spinner("Generating answer..."):
                conversation = st.session_state['conversation']
                response = conversation({"query": user_question})
                answer = response["result"]
                
                st.session_state["question"] = user_question
                st.session_state["answer"] = answer
                st.session_state.setdefault("history", []).append((user_question, answer))
        
        st.write("### Answer:")
        st.write(answer)

if __name__ == '__main__':
    main()
