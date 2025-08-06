import os
from pathlib import Path

from tavily import TavilyClient
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


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API"))


def web_search(query):
    """
    Function for web search using Tavily API

    Args:
        query (str): Search query

    Returns:
        list: Search results
    """
    try:
        response = tavily_client.search(max_results=5, query=query)
        return response
    except Exception as e:
        print(f"An error occurred during web search: {str(e)}")
        return []


class Model(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ReAct(Chat, tools=[web_search])

    def forward(self, question: str):
        return self.respond(question=question)
    
Tool_calling_LLM = Model()


corpus = []

folder_path = "./company_map"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()  # Remove any leading/trailing whitespace
            if text:  # Only add non-empty strings
                corpus.append(str(text))  # Ensure it's a string
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")

topk_docs_to_retrieve = 1

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)

class RAG(dspy.Signature):
    context: str = dspy.InputField(desc="Context provided to help answer the question")
    question: str = dspy.InputField(desc="Questions asked by the user")
    response: str = dspy.OutputField(desc="Response to the question")

class Model(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrive = dspy.Retrieve(k=5)
        self.respond = dspy.ChainOfThought(RAG)

    def forward(self, question: str):
        context = search(question).passages
        return self.respond(context=context, question=question)
    
rag = Model()


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Therapy LLM", "RAG LLM", "Tool Calling LLM"])

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = page

# Clear messages if page changes
if st.session_state.current_page != page:
    st.session_state.messages = []
    st.session_state.current_page = page

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
            elif page == "Tool Calling LLM":
                response = Tool_calling_LLM(question=prompt)
                st.write(response.response)
            elif page == "RAG LLM":
                response = rag(question=prompt)
                st.write(response.response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.response})


# Main content area
if page == "Therapy LLM":
    st.title("🧠 Domain specific Therapy LLM")
    st.write("Welcome to the Therapy LLM. I'm here to help you with your emotional well-being.")
    display_chat_interface()
elif page == "RAG LLM":
    st.title("📚 Company specific RAG LLM")
    st.write("Welcome to the RAG LLM. I can retrieve information about the made up company NordicTech Solutions.")
    display_chat_interface()
else:  # Tool Calling LLM
    st.title("🛠️ Tool Calling LLM")
    st.write("Welcome to the Tool Calling LLM. I have a tool to search the web!")
    display_chat_interface()
