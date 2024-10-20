import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
import os
from dotenv import load_dotenv

# Load environment variables for OpenAI API key and valid credentials
load_dotenv()

# Get the valid credentials from the environment
VALID_CREDENTIALS = os.getenv("VALID_CREDENTIALS", "")
VALID_CREDENTIALS = dict(cred.split(":") for cred in VALID_CREDENTIALS.split(","))

# Load your CSV file with FAQs
loader = CSVLoader(file_path="FairPrice Online.csv", encoding="utf-8")

# Set up the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create the VectorstoreIndex using LangChain (this helps in indexing the data properly)
index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader])

# Use a valid model (e.g., gpt-4, gpt-3.5-turbo)
llm = ChatOpenAI(model_name="gpt-4")

# Create the retrieval-based QA chain using RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="input")

# Function to generate chatbot response from CSV-based data
def get_chatbot_response(user_input):
    response = chain({"input": user_input})
    return response['result']

# Function to verify login credentials
def check_login(email, password):
    return VALID_CREDENTIALS.get(email) == password

# Page routing
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# Avoid redundant rerun calls
def rerun():
    st.rerun()

# Set up page navigation
if st.session_state["page"] == "login":
    # Login page
    st.set_page_config(page_title="Login - FairPrice FAQ Chatbot", layout="centered")
    st.title("Login to FairPrice FAQ Chatbot 🤖")

    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")

    if st.button("Login"):
        if check_login(email, password):
            st.session_state["logged_in"] = True
            st.session_state["page"] = "chatbot"
            st.success("Login successful! Redirecting to chatbot...")
            rerun()
        else:
            st.error("Invalid email or password. Please try again.")

elif st.session_state["page"] == "chatbot":
    if not st.session_state.get("logged_in"):
        st.session_state["page"] = "login"
        rerun()

    # Chatbot page
    st.set_page_config(page_title="FairPrice FAQ Chatbot", page_icon="🤖", layout="centered")
    st.title("FairPrice FAQ Chatbot 🤖")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 6])  # Adjust column widths as needed

    with col1:
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["page"] = "login"
            rerun()  # Redirect back to the login page after logout

    with col2:
        st.markdown("Ask any questions about FairPrice app features, and we'll help you!")

    # Initialize the session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Chat history display
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user query with chat input field
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Store user query in the session state
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Get chatbot response
        response = get_chatbot_response(user_input)

        # Store bot response in session state
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Display both user question and bot response in the chat interface
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            st.markdown(response)
