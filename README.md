# 🤖 Salesforce Q&A Assistant (RAG-based)

This is a **Retrieval-Augmented Generation (RAG)** assistant that answers questions strictly related to **Salesforce**. It combines:

- 🔍 FAISS-based vector search over official Salesforce docs
- 🧠 Local LLM via DeepInfra (`mistralai/Mistral-7B-Instruct-v0.2`)
- 🌐 Fallback web search using DuckDuckGo (only for Salesforce queries)
- 💻 Deployed on **Streamlit Cloud** (100% free-tier compatible)

---

## 🚀 Live Demo

👉 [Click here to try the app on Streamlit](https://salesforce-agent-aleg6t54qv8nx5c3wvsuul.streamlit.app/)])

---

## 🧱 Architecture


User Query
│
▼
[FAISS Vector Search] ← Index of Salesforce Docs
│ ▲
│ └── Chunking + Embedding (MiniLM-L6-v2)
│
[If no relevant context]
│
▼
[DuckDuckGo Fallback] ← Custom throttled web search
│
▼
[Mistral-7B-Instruct (via DeepInfra)]
│
▼
Answer (only if Salesforce-relevant)


---

## 📦 Docker

You can run the app with Docker as well:


docker pull shuvam1998/salesforce-agent:latest
docker run -p 8501:8501 -e DEEPINFRA_API_TOKEN=your_key shuvam1998/salesforce-agent
 View on Docker Hub: shuvam1998/salesforce-agent





---

## 📁 Directory Structure

├── app.py # Streamlit frontend
├── .streamlit/secrets.toml # API key for DeepInfra (never commit this)
├── src/
│ ├── rag_pipeline.py # RAG logic (retrieval + fallback)
│ └── utils.py # DuckDuckGo throttled search class
├── faiss_index/ # FAISS index (ignored from GitHub)
├── requirements.txt
└── README.md



---

## 🧪 Example Questions (Try these)

| ✅ Relevant                             | ❌ Blocked |
|----------------------------------------|------------|
| What is a Salesforce object?           | Who is the president of India? |
| How to create a Lightning Component?   | What is the weather in Delhi? |
| What is Apex in Salesforce?            | Define photosynthesis |
| How does SOQL work?                    | Who won the FIFA World Cup? |

---

## 🔧 Setup Instructions

### 1. Clone the repo

git clone https://github.com/shuvam1998/salesforce-agent.git
cd salesforce-agent



2. Create virtual environment

python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

3. Install dependencies

pip install -r requirements.txt

4. Add API Key
Create a .streamlit/secrets.toml:

toml
DEEPINFRA_API_TOKEN = "your_deepinfra_key"
📡 Indexing Docs (Run Once Locally)

python src/rag_pipeline.py
This saves a FAISS index to faiss_index/, which must be uploaded to your HF repo.

📁 FAISS files are hosted here:
🔗 HuggingFace Repo

Only index.faiss and index.pkl are stored — no model weights or code.

☁️ Streamlit Cloud Deployment
Fork or clone this repo.

Go to Streamlit Cloud and connect GitHub.

Set secret DEEPINFRA_API_TOKEN.

Set entrypoint to app.py.

Click Deploy.

📜 License
MIT License. Free to use with attribution. Give a ⭐ if helpful!
























