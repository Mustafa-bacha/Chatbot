import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
import os
from dotenv import load_dotenv

# Load environment variables for OpenAI API key
#_ = load_dotenv()
_ = st.secrets['OPENAI_API_KEY']


# Load your CSV file with FAQs
loader = CSVLoader(file_path="combined_data.csv", encoding="utf-8")

# Set up the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create the VectorstoreIndex using Langchain
index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader])

# Create the retrieval-based QA chain
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

# Function to generate chatbot response
def get_chatbot_response(user_input):
    response = chain({"question": user_input})
    return response['result']

# Set up Streamlit page config
st.set_page_config(page_title="FairPrice FAQ Chatbot", page_icon="🤖", layout="centered")

# Display the chatbot icon and heading
st.title("FairPrice FAQ Chatbot 🤖")
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