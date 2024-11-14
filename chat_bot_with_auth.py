import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage 
import pyrebase
from google.cloud import firestore
import json
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load environment variables
load_dotenv()

# Firebase configuration
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firestore.Client()

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

def calculate_similarity(query_embedding, doc_embeddings):
    return cosine_similarity([query_embedding], doc_embeddings).flatten()

def generate_dynamic_out_of_context_message(summary):
    return (
        f"The sources provided are technical documents, primarily {summary}. "
        "They contain information related to the document's specific content, but no relevant details were found for your query."
    )

def register_user(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return user
    except Exception as e:
        return None

def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user
    except:
        return None

def save_conversation(user_id, question, answer):
    conversation_ref = db.collection("conversations").document(user_id)
    
    # Prepare conversation data with a placeholder for timestamp
    conversation_data = {
        "question": question,
        "answer": answer,
        "timestamp": None  # Set as None initially
    }

    # Step 1: Add the conversation without the timestamp
    if not conversation_ref.get().exists:
        conversation_ref.set({"conversations": []})
    
    # Use ArrayUnion to append without timestamp
    conversation_ref.update({
        "conversations": firestore.ArrayUnion([conversation_data])
    })

    # Step 2: Update the timestamp separately with a Firestore transaction
    conversation_data_with_timestamp = {
        "conversations": firestore.ArrayUnion([
            {
                "question": question,
                "answer": answer,
                "timestamp": firestore.SERVER_TIMESTAMP  # Add timestamp now
            }
        ])
    }

    # Replace the placeholder entry with the one containing SERVER_TIMESTAMP
    # db.run_transaction(lambda transaction: transaction.update(conversation_ref, conversation_data_with_timestamp))


def get_conversations(user_id):
    doc = db.collection("conversations").document(user_id).get()
    if doc.exists:
        return doc.to_dict().get("conversations", [])
    return []

def logout():
    if 'user' in st.session_state:
        del st.session_state['user']
        st.session_state['login_success'] = False
        st.success("You have been logged out.")
        st.rerun()

# Main Application
def main():
    st.header("Chat with your PDFs :books:")

    # Check if the user is already logged in
    if 'user' not in st.session_state:
        st.subheader("Login / Sign Up")
        
        # Tabs for login and sign-up
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        with login_tab:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login",key="login_button"):
                user = login_user(email, password)
                if user:
                    st.session_state['user'] = user
                    st.session_state['login_success'] = True
                    st.success("Logged in successfully")
                else:
                    st.error("Invalid credentials")

        with signup_tab:
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Sign Up",key="signup_button"):
                user = register_user(email, password)
                if user:
                    st.success("Account created successfully, please log in.")
                else:
                    st.error("Failed to create account, try again.")
        
        return  # Exit the function here if the user is not logged in

    # Trigger rerun after login is successful
    if 'login_success' in st.session_state and st.session_state['login_success']:
        st.session_state['login_success'] = False
        st.rerun()

    # Main Chat Screen for Authenticated Users
    user_id = st.session_state['user']['localId']
    
    # Sidebar - PDF Upload and Process
    with st.sidebar:
        if st.button("Logout", key="logout_button"):
            logout()

        st.subheader("Your Documents:")
        pdf_docs = st.file_uploader("Upload PDFs and click on 'Process'", accept_multiple_files=True)

        if st.button("Process New PDFs", key="process_pdfs_button"):
            if pdf_docs:
                with st.spinner("Processing new documents..."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)
                    summary = generate_summary(raw_text)
                    vector_store = store_chunks_and_create_vector_store(text_chunks)
                    st.session_state['vector_store'] = vector_store
                    st.session_state['summary'] = summary
                    st.success("Documents processed and added to vector store. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF document.")

        # # Conversation History
        # st.subheader("Past Conversations:")
        # conversations = get_conversations(user_id)
        
        # # Display clickable questions in history
        # for i, convo in enumerate(conversations):
        # # Use the conversation question as part of the key to ensure it's unique
        # # if st.button(convo['question'], key=f"convo_button_{i}"):  # Use i to make the key unique
        #     st.session_state['question_input'] = convo['question']
        #     st.session_state['answer_output'] = convo['answer']


    # Main Chat Interface
    user_question = st.text_input("Ask a question from the uploaded documents:", value=st.session_state.get('question_input', ''))
    
    if 'vector_store' in st.session_state and user_question:
        with st.spinner("Generating answer..."):
            vector_store = st.session_state['vector_store']
            summary = st.session_state.get("summary", "a Document")
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            query_embedding = embeddings.embed_query(user_question)
            
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(user_question)
            
            doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
            similarity_scores = calculate_similarity(query_embedding, doc_embeddings)
            
            relevant_docs = sorted(
                [(doc, score) for doc, score in zip(retrieved_docs, similarity_scores) if score >= SIMILARITY_THRESHOLD],
                key=lambda x: x[1],
                reverse=True
            )

            if relevant_docs:
                top_chunks = "\n\n".join([doc.page_content for doc, _ in relevant_docs])
                llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
                user_message = HumanMessage(content=f"Answer the question '{user_question}' based on the following information:\n\n{top_chunks}")
                response = llm([user_message])
                answer = response.content
            else:
                answer = generate_dynamic_out_of_context_message(summary)

            st.session_state['answer_output'] = answer

            # Display the answer
            st.write("### Answer:")
            st.write(answer)

            # Save conversation
            save_conversation(user_id, user_question, answer)

if __name__ == '__main__':
    main()
