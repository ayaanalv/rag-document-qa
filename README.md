# RAG Document Q&A

A retrieval-augmented generation (RAG) pipeline that answers natural-language questions about PDF documents, using a fully local LLM stack via [Ollama](https://ollama.com) — no API keys, no cloud costs.

## What it does

Point it at a folder of PDFs, ask a question in plain English, and it:
1. Finds the most relevant sections of the PDFs using semantic search (not just keyword matching)
2. Hands only those relevant sections to a local LLM
3. Returns an answer grounded in the actual document content, with source attribution per chunk

This means the model can only answer using what's actually in your documents — it won't make things up from its own training data, and it will say so if the answer isn't present.

## Tech stack

- **Python 3**
- **[Ollama](https://ollama.com)** — runs both the embedding model (`nomic-embed-text`) and the chat model (`llama3.2`) locally
- **pypdf** — extracts text from PDF files
- **numpy** — computes cosine similarity between embedding vectors

## How it works

1. **Chunking** — each PDF is split into overlapping ~1000-character chunks, so related content doesn't get cut off mid-thought at chunk boundaries
2. **Embedding** — every chunk is converted into a vector using `nomic-embed-text`, capturing semantic meaning rather than exact word matches
3. **Retrieval** — when a question comes in, it's embedded the same way, and compared against every chunk via cosine similarity; the top 5 most relevant chunks are selected
4. **Generation** — the question and retrieved chunks are passed to `llama3.2`, which is instructed to answer using only that context

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/download) installed

```bash
# Clone the repo
git clone https://github.com/ayaanalv/rag-document-qa.git
cd rag-document-qa

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull the required Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

Add one or more PDFs to the `documents/` folder, then run:

```bash
python rag_pdf.py
```

You'll be prompted to type a question. The script will print which chunks were retrieved (with their source file) and the generated answer.

## Project structure

```
rag-document-qa/
├── documents/       # put your PDFs here (not tracked by git)
├── rag_pdf.py        # main pipeline: PDF loading, embedding, retrieval, generation
├── rag_ollama.py      # early prototype using hardcoded text instead of real PDFs — kept for reference, not the active version
└── requirements.txt
```

## Known limitations

- Chunk overlap can occasionally cause the same source content to be retrieved twice, which the model may present as two separate mentions
- No handling for scanned/image-only PDFs (no OCR step)
- Embedding every chunk happens on each run rather than being cached, so startup time scales with document size

## Background

Built to explore retrieval-augmented generation and local LLM tooling, motivated by internship postings requiring hands-on RAG/GenAI experience.