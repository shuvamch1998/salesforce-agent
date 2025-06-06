import os
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import ThrottledDuckDuckGoSearch

# Salesforce documentation URLs (only used during local indexing)
urls = [
    "https://developer.salesforce.com/docs/platform",
    "https://trailhead.salesforce.com/en/content/learn/modules"
]

# Monkey patch for setting custom headers during doc loading
def custom_requests_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({"User-Agent": "Mozilla/5.0"})
    return requests.get(url, headers=headers, **kwargs)

# STEP 1: Local run-once script to build FAISS index

def load_and_index_salesforce_docs(urls):
    print("\n🔍 [INFO] Loading Salesforce docs...")
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loader.requests_get = custom_requests_get
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print("[INFO] Embedding chunks using HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ [INFO] FAISS index saved to `faiss_index/`")

# STEP 2: Load FAISS index at runtime

def load_retriever():
    print("[INFO] Loading FAISS index from huggingface.co manually uploaded files...")
    os.makedirs("faiss_index", exist_ok=True)
    base_url = "https://huggingface.co/shuvam1998/salesforce-agent-faiss/resolve/main"

    for filename in ["index.faiss", "index.pkl"]:
        response = requests.get(f"{base_url}/{filename}")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {filename} from HF")
        with open(os.path.join("faiss_index", filename), "wb") as f:
            f.write(response.content)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()

# STEP 3: Combined context + fallback logic
search = ThrottledDuckDuckGoSearch()

def answer_with_context(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    if not context.strip():
        if "salesforce" in query.lower():
            print("🧭 [INFO] No FAISS match, searching DuckDuckGo...")
            context = search.run(query)
            prompt = f"""
You are a Salesforce expert. Use the web search result below to answer the user's question.
Do NOT answer if the search result is not clearly related to Salesforce.

Context:
{context}

Question: {query}
"""
            return llm.invoke(prompt)
        else:
            return "⚠️ This assistant only answers questions related to Salesforce."

    # If context is found in FAISS
    prompt = f"""
You are a helpful Salesforce assistant. Use the context below to answer the user's question.
Do NOT answer if the question is unrelated to Salesforce.

Context:
{context}

Question: {query}
"""
    return llm.invoke(prompt)

# Optional: Run this once locally to build index
# if __name__ == "__main__":
#     load_and_index_salesforce_docs(urls)
