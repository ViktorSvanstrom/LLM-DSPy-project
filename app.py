import os
from pathlib import Path

import dspy
import streamlit as st
from dotenv import load_dotenv

# Streamlit app configuration - must be the first Streamlit command
st.set_page_config(page_title="LLM Assistant", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_dspy():
    # Initialize the language model
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=1)
    dspy.configure(lm=lm)
    return lm

# Initialize DSPy
lm = initialize_dspy()

# Create the program
class Chat(dspy.Signature):
    "You are a helpful assistant."

    question: str = dspy.InputField(desc="Questions asked by the user")
    response: str = dspy.OutputField(desc="Response to the question")


class Model(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(Chat)

    def forward(self, question: str):
        return self.respond(question=question)


# Initialize the Therapy LLM
therapy_LLM = Model()
therapy_LLM.load("./dspy_programs/counseling_program.json")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Therapy LLM", "RAG LLM", "Tool Calling LLM"])

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to display chat interface
def display_chat_interface():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from LLM
        with st.chat_message("assistant"):
            if page == "Therapy LLM":
                response = therapy_LLM(question=prompt)
                st.write(response.response)
            else:
                st.write("This LLM is not yet implemented.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.response if page == "Therapy LLM" else "This LLM is not yet implemented."})


# Main content area
if page == "Therapy LLM":
    st.title("🧠 Therapy LLM")
    st.write("Welcome to the Therapy LLM. I'm here to help you with your emotional well-being.")
    display_chat_interface()
elif page == "RAG LLM":
    st.title("📚 RAG LLM")
    st.write("Welcome to the RAG LLM. This feature is coming soon!")
    display_chat_interface()
else:  # Tool Calling LLM
    st.title("🛠️ Tool Calling LLM")
    st.write("Welcome to the Tool Calling LLM. This feature is coming soon!")
    display_chat_interface()
