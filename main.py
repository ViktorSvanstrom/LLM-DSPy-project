import os

import dspy
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM("gpt-4o", api_key=openai_api_key)

local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=local_lm)



if __name__ == "__main__":
    pass
