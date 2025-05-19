import os

import dspy
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure the language model with required parameters
lm = dspy.LM("gpt-4o", api_key=openai_api_key, temperature=0.7, max_tokens=1000)

# Configure DSPy with the language model
dspy.configure(lm=lm)

local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=local_lm)


if __name__ == "__main__":
    pass
