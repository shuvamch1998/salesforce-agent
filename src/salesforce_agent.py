from langchain_community.llms import DeepInfra
from rag_pipeline import load_retriever, answer_with_context
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load DeepInfra API key
api_token = os.getenv("DEEPINFRA_API_TOKEN")

# Load the LLM from DeepInfra
llm = DeepInfra(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    deepinfra_api_token=api_token
)

# Load FAISS retriever
retriever = load_retriever()

# Take user query
query = input("Ask a Salesforce-related question: ").strip()

# Enforce Salesforce-only prompt within this file itself
custom_prompt = f"""
You are a helpful assistant specialized in Salesforce-related topics only.

Only use the following context to answer the user's question. If the question is unrelated to Salesforce or cannot be answered using the context, respond with "I'm sorry, I can only answer questions about Salesforce topics."

Context and retrieved content will follow.
Question: {query}
"""

print("\n🔍 Grounded Answer:\n")
print(answer_with_context(custom_prompt, retriever, llm))
