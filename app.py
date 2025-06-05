import os
import sys
from dotenv import load_dotenv
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from langchain_community.llms import DeepInfra
from rag_pipeline import load_retriever, answer_with_context



load_dotenv()

st.set_page_config(page_title="Salesforce RAG Agent", layout="centered")

st.title("🤖 Salesforce Q&A Assistant")
query = st.text_input("Ask your Salesforce-related question:")

if query:
    with st.spinner("Thinking..."):
        retriever = load_retriever()
        llm = DeepInfra(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            api_token=os.getenv("DEEPINFRA_API_TOKEN")
        )
        answer = answer_with_context(query, retriever, llm)
        st.markdown(f"**Answer:**\n\n{answer}")
