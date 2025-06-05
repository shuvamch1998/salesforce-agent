import os
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import ThrottledDuckDuckGoSearch

# Salesforce documentation URLs (used only for local indexing)
urls = [
    "https://developer.salesforce.com/docs/platform",
    "https://trailhead.salesforce.com/en/content/learn/modules"
]

# Custom User-Agent to avoid blocking
def custom_requests_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({"User-Agent": "Mozilla/5.0"})
    return requests.get(url, headers=headers, **kwargs)

# STEP 1: Index Salesforce docs (optional)
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
    vectorstore.save_local("faiss_files")
    print("✅ [INFO] FAISS index saved to `faiss_files/`")

# STEP 2: Load from Hugging Face repo
def load_retriever():
    print("[INFO] Downloading FAISS index from Hugging Face...")

    base_url = "https://huggingface.co/shuvamch1998/salesforce-rag-faiss/resolve/main"
    os.makedirs("faiss_files", exist_ok=True)

    for filename in ["index.faiss", "index.pkl"]:
        url = f"{base_url}/{filename}"
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}: {r.status_code}")
        with open(os.path.join("faiss_files", filename), "wb") as f:
            f.write(r.content)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_files", embeddings, allow_dangerous_deserialization=True).as_retriever()

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

# Optional run locally
# if __name__ == "__main__":
#     load_and_index_salesforce_docs(urls)
