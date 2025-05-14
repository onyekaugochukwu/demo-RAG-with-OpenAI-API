import streamlit as st
import openai
from datetime import datetime
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
import tempfile

# Set page configuration
st.set_page_config(page_title="Document Chat App", page_icon="📄")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = "xxxxxxx"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def initialize_openai_client():
    """Initialize the OpenAI client with the provided API key."""
    if st.session_state.api_key:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        return client
    return None

def process_file(uploaded_file):
    """Process the uploaded file and create vector store."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Load document based on file type
    if uploaded_file.name.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
    elif uploaded_file.name.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    os.unlink(file_path)  # Clean up temporary file
    return vector_store

def get_relevant_context(query):
    """Retrieve relevant context from vector store."""
    if st.session_state.vector_store is None:
        return ""
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

def get_response(messages):
    """Get a response from OpenAI API based on conversation history."""
    try:
        client = initialize_openai_client()
        if not client:
            return {"role": "assistant", "content": "Please enter a valid API key."}
        
        # Get relevant context from the document
        user_message = messages[-1]["content"]
        context = get_relevant_context(user_message)
        
        # Add context to the system message
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer questions: {context}"},
        ] + messages
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Updated to use GPT-4 Turbo
            messages=messages,
            temperature=0.7,
        )
        
        return {"role": "assistant", "content": response.choices[0].message.content}
    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}

def main():
    st.title("📄 Document Chat App")
    
    # API key input and file upload in sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            st.session_state.api_key = api_key
        
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            # Show file details
            st.write("File Details:")
            st.write(f"- Name: {uploaded_file.name}")
            st.write(f"- Size: {uploaded_file.size / 1024:.2f} KB")
            st.write(f"- Type: {uploaded_file.type}")
            
            if st.session_state.api_key:
                with st.spinner("Processing document..."):
                    st.session_state.vector_store = process_file(uploaded_file)
                    st.success("Document processed successfully!")
        
        # Add clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
        
        st.divider()
        st.write("Made with ❤️ using Streamlit")
        st.write("© " + str(datetime.now().year))
    
    # Display chat interface
    if st.session_state.vector_store is None:
        st.info("Please upload a document to start chatting.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type a message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get API response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages_for_api = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + st.session_state.messages
                
                response = get_response(messages_for_api)
                st.write(response["content"])
        
        # Add assistant response to chat history
        st.session_state.messages.append(response)

if __name__ == "__main__":
    main()