# Medical-Chatbot-RAG

Medical-Chatbot-RAG is a Flask web application that answers medical questions using Retrieval-Augmented Generation (RAG).
It retrieves relevant passages from medical PDF documents stored in Pinecone, then uses a Groq-hosted LLM to generate concise, context-grounded responses.

## What This Project Is Used For

- Build a chatbot that answers medical queries from your own PDF knowledge base.
- Reduce hallucinations by forcing answers to come from retrieved document context.
- Provide a simple web chat UI for testing healthcare Q and A workflows.

## Tech Stack

- Flask for the web application
- LangChain for retrieval and RAG orchestration
- Pinecone as the vector database
- Groq LLM for answer generation
- Sentence Transformers embeddings model: all-MiniLM-L6-v2

## Prerequisites

- Python 3.10+
- Pinecone account and API key
- Groq API key

## Step-by-Step: Download and Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MotiranjanRath/Medical-Chatbot-RAG.git
cd Medical-Chatbot-RAG
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails due to the last invalid line in requirements.txt, delete the last line and run the install command again.

### 4. Create environment variables

Create a .env file in the project root:

```bash
cat > .env << 'EOF'
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
EOF
```

### 5. Add medical PDFs

Place your PDF files inside the data directory:

```bash
ls data
```

### 6. Build the Pinecone vector index

Before running indexing, update the PDF path in Vstore_index.py to your local data path.

Recommended value:

```python
extracted_data = extract_text_from_pdf("data")
```

Then run:

```bash
python Vstore_index.py
```

### 7. Start the application

```bash
python app.py
```

The app starts on:

- http://localhost:8000

## How It Works

1. PDFs are loaded and chunked.
2. Chunks are embedded with Sentence Transformers.
3. Embeddings are stored in Pinecone.
4. User question is sent to retriever.
5. Retrieved context and question are passed to Groq LLM.
6. Answer is returned in the chat UI.

## Project Structure

```text
.
├── app.py
├── Vstore_index.py
├── data/
├── src/
│   ├── helper.py
│   └── prompt.py
├── static/
├── templates/
└── requirements.txt
```