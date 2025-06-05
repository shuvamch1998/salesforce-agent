import streamlit as st
from rag_pipeline import load_retriever, answer_with_context
from langchain_community.llms import DeepInfra
import os
from dotenv import load_dotenv

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
