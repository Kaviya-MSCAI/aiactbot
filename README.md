# AIActBot — EU AI Act Compliance Assistant

Ask plain English questions about the EU AI Act and get cited, accurate answers instantly.

Built as a portfolio project demonstrating RAG, LLM engineering, and EU AI compliance knowledge for the Irish AI job market.

---

## What it does

- Upload the official EU AI Act PDF
- Ask any compliance question in plain English
- Get answers with exact page references and article citations
- Multi-turn conversation — remembers context across questions

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Llama 3.2 (via Ollama) |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Document | EU AI Act 2024 (Official PDF) |

---

## How to run locally

### 1. Clone the repo
```bash
git clone https://github.com/Kaviya-MSCAI/aiactbot.git
cd aiactbot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull model
Download Ollama from [ollama.com](https://ollama.com) then run:
```bash
ollama pull llama3.2
```

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Upload the EU AI Act PDF
Download from [artificialintelligenceact.eu](https://artificialintelligenceact.eu) and upload in the sidebar.

---

## How it works
```
PDF Upload
    │
    ▼
PyPDF extracts text from all pages
    │
    ▼
RecursiveCharacterTextSplitter creates 1000-char chunks
    │
    ▼
Sentence Transformers embed each chunk into vectors
    │
    ▼
ChromaDB stores vectors locally
    │
    ▼
User question → embedded → top 5 similar chunks retrieved
    │
    ▼
Llama 3.2 generates plain English answer with citations
```

---

## Skills demonstrated

- Python
- LangChain
- ChromaDB
- Retrieval Augmented Generation (RAG)
- Vector Databases
- Prompt Engineering
- Streamlit
- Ollama / Local LLMs
- EU AI Act domain knowledge

---

## Disclaimer

AIActBot is for informational purposes only and does not constitute legal advice.
Always consult a qualified legal professional for compliance decisions.

---

Built by Kaviya Rajavel
