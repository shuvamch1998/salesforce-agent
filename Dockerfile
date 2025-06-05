# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and FAISS index
COPY ./src ./src
COPY ./faiss_index ./faiss_index
COPY .env .env

# Set environment variables (for LLM fallback and custom user-agent)
ENV USER_AGENT="Mozilla/5.0"

# Entry point
CMD ["python", "src/salesforce_agent.py"]

