import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import json

# Load environment variables (API keys, etc.) from a .env file
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
            text += page.extract_text()
    
    # Limit the text to the first 50,000 characters to avoid tokenization issues
    text = text[:50000]
    return text

# Function to summarize the text using OpenAI's Chat model
def summarize_text(text):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key is not set.")
        return text  

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.7)
    summaries = []
    text_splitter = TokenTextSplitter(chunk_size=5000, chunk_overlap=500)
    text_chunks = text_splitter.split_text(text)

    # Summarize each chunk individually to manage token limits
    for chunk in text_chunks:
        summary = llm.predict(f"Please summarize the following text:\n\n{chunk}")
        summaries.append(summary)

    # Combine all summaries into a single string
    final_summary = " ".join(summaries)
    return final_summary

# Function to split text into smaller chunks for vector embedding
def get_text_chunks(text):
    text_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)

# Function to store text chunks in a JSON file and create a FAISS vector store
def ingest_documents(text_chunks):
    with open(CHUNKS_PATH, "w") as f:
        json.dump(text_chunks, f)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# Function to load existing vector store from file if available
def get_existing_vector_store():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r") as f:
            text_chunks = json.load(f)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        return vector_store
    return None

# Function to create a conversation chain with custom logic for handling no document matches
def get_conversation_chain(vector_store):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=openai_api_key, temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom chain to handle cases where no relevant documents are found
    class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
        def _call(self, inputs):
            query = inputs["question"]
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                # If no documents match, generate a general answer instead
                llm_response = llm.predict(f"Answer this question based on general knowledge: {query}")
                return {"answer": llm_response}
            else:
                return super()._call(inputs)

    return CustomConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# Session state variables initialization
if "is_from_history" not in st.session_state:
    st.session_state["is_from_history"] = False  # Tracks if the input is from a history selection
if "new_chat" not in st.session_state:
    st.session_state["new_chat"] = False  # Tracks if the new chat button is clicked

# Main app function
def main():
    st.header("Chat with multiple PDFs :books:")
    
    # Sidebar for uploading PDFs and history navigation
    with st.sidebar:
        st.subheader("Your documents here:")
        pdf_docs = st.file_uploader("Upload PDFs and click on 'Process'", accept_multiple_files=True)
        
        # Show the "New Chat" button only if there's a question and an answer
        if "question" in st.session_state and "answer" in st.session_state:
            if st.session_state["question"] and st.session_state["answer"]:
                # Display "New Chat" button to reset the session
                if st.button("🗨️ New Chat"):
                    # Clear the selected history, reset flags, and clear input fields
                    if "selected_history" in st.session_state:
                        del st.session_state["selected_history"]
                    st.session_state["is_from_history"] = False
                    st.session_state["new_chat"] = True
                    # Reset the question and answer fields
                    st.session_state["question"] = ""
                    st.session_state["answer"] = ""

        # Process new PDFs, extract and embed their contents
        if st.button("Process New PDFs"):
            if pdf_docs:
                with st.spinner("Processing new documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    summarized_text = summarize_text(raw_text)
                    text_chunks = get_text_chunks(summarized_text)
                    vector_store = ingest_documents(text_chunks)
                    conversation = get_conversation_chain(vector_store)
                    st.session_state['conversation'] = conversation
                    
                    # Clear selected history and reset flags after processing new PDFs
                    if "selected_history" in st.session_state:
                        del st.session_state["selected_history"]
                    st.session_state["is_from_history"] = False

                    st.success("Documents processed and added to vector store. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF document.")

        # Load existing document embeddings and initialize a conversation chain
        if st.button("Load Existing Documents"):
            vector_store = get_existing_vector_store()
            if vector_store:
                conversation = get_conversation_chain(vector_store)
                st.session_state['conversation'] = conversation
                
                # Clear selected history and reset flags after loading existing documents
                if "selected_history" in st.session_state:
                    del st.session_state["selected_history"]
                st.session_state["is_from_history"] = False

                st.success("Loaded existing vector store. You can now ask questions.")
            else:
                st.warning("No existing documents found. Please upload and process new PDFs first.")
        
        # Display chat history in sidebar
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(st.session_state.get("history", [])):
            # Selecting a previous question from the history
            if st.button(f"Q{i+1}: {question[:30]}..."):
                st.session_state["selected_history"] = (question, answer)
                st.session_state["is_from_history"] = True  # Mark that input is from history selection

    # Display the selected question from history in the input box, if applicable
    if "selected_history" in st.session_state and not st.session_state.get("new_chat", False):
        selected_question, selected_answer = st.session_state["selected_history"]
        user_question = st.text_input("Ask a question from the uploaded documents:", selected_question)
    else:
        # If not from history, allow user to input a new question
        user_question = st.text_input("Ask a question from the uploaded documents:")
        st.session_state["new_chat"] = False  # Reset new chat flag after input box is updated

    # Check if there's a conversation chain in session and if a question was entered
    if 'conversation' in st.session_state and user_question:
        # Check if the current input is from a history selection
        if st.session_state["is_from_history"]:
            # Display the answer directly from the selected history
            answer = selected_answer
            st.session_state["is_from_history"] = False  # Reset flag after displaying history item
        else:
            # Generate a response for the new question
            with st.spinner("Generating answer..."):
                conversation = st.session_state['conversation']
                response = conversation({"question": user_question})
                
                # Extract the answer and display it
                answer = response["answer"]
                
                # Save the current question and answer to session state for later
                st.session_state["question"] = user_question
                st.session_state["answer"] = answer
                
                # Append new Q&A to history
                st.session_state.setdefault("history", []).append((user_question, answer))
        
        # Display the answer below the input
        st.write("### Answer:")
        st.write(answer)

if __name__ == '__main__':
    main()

