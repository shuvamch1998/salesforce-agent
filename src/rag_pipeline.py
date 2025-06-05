import os
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download
from utils import ThrottledDuckDuckGoSearch

# Salesforce documentation URLs (used for offline generation only)
urls = [
    "https://developer.salesforce.com/docs/platform",
    "https://trailhead.salesforce.com/en/content/learn/modules"
]

# Monkey patch to override default headers
def custom_requests_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({"User-Agent": "Mozilla/5.0"})
    return requests.get(url, headers=headers, **kwargs)

# STEP 1: Load and index Salesforce docs locally (optional run-once script)
def load_and_index_salesforce_docs(urls):
    print("\n🔍 [INFO] Loading Salesforce docs...")

    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loader.requests_get = custom_requests_get  # override here
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print("[INFO] Embedding chunks using HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_files")
    print("✅ [INFO] FAISS index saved to `faiss_files/`")

# STEP 2: Load FAISS index from Hugging Face Hub
def load_retriever():
    print("[INFO] Downloading FAISS index from Hugging Face Hub...")
    
    # Update with your HF repo ID
    repo_id = "shuvamch1998/salesforce-rag-faiss"

    faiss_path = hf_hub_download(repo_id=repo_id, filename="index.faiss")
    pkl_path = hf_hub_download(repo_id=repo_id, filename="index.pkl")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS expects the folder path, so we strip off the file
    index_folder = os.path.dirname(faiss_path)
    return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True).as_retriever()

# STEP 3: Answer with retrieved context or fallback
search = ThrottledDuckDuckGoSearch()

def answer_with_context(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    if not context.strip():
        print("🧭 [INFO] No relevant FAISS context found, using DuckDuckGo fallback...\n")
        context = search.run(query)

    prompt = f"""
    You are a helpful Salesforce assistant. Use the context below to answer the user's question as accurately as possible.

    Context:
    {context}

    Question: {query}
    """
    return llm.invoke(prompt)

# Uncomment and run locally (not for Streamlit Cloud)
 #if __name__ == "__main__":
  #   load_and_index_salesforce_docs(urls)
