import os
from langchain_ollama import OllamaLLM

# A test script to ensure llama setup is working from python
llama_base_url = os.environ["LLAMA_BASE_URL"]

model = ChatOllama(
        model="llama3", 
        base_url=llama_base_url, 
        temperature=0, 
        verbose=True
        # headers=headers # any headers needed
    )

print(model.invoke("Come up with 10 food based names for dogs"))