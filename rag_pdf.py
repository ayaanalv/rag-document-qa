"""
RAG pipeline that reads real PDFs from a folder, instead of hardcoded chunks.
Run this on YOUR machine with Ollama running.
"""
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

DOCS_FOLDER = "documents"   # put your PDFs in a folder with this name
CHUNK_SIZE = 500            # characters per chunk
CHUNK_OVERLAP = 50          # characters shared between consecutive chunks

# ---- PHASE 1: SETUP ----

def load_and_chunk_pdfs(folder):
    """Read every PDF in `folder`, split each into overlapping text chunks."""
    all_chunks = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder, filename)
        reader = PdfReader(filepath)

        # Pull all text out of the PDF, page by page
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # Split into overlapping chunks. Overlap matters: if a sentence gets
        # cut in half at a chunk boundary, the overlap gives the next chunk
        # enough surrounding context to still make sense on its own.
        start = 0
        while start < len(full_text):
            end = start + CHUNK_SIZE
            chunk_text = full_text[start:end].strip()
            if chunk_text:  # skip empty chunks (e.g. blank pages)
                all_chunks.append({"text": chunk_text, "source": filename})
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return all_chunks

chunk_records = load_and_chunk_pdfs(DOCS_FOLDER)
chunks = [c["text"] for c in chunk_records]

if not chunks:
    raise SystemExit(
        f"No PDFs found in '{DOCS_FOLDER}/'. "
        f"Create that folder next to this script and put at least one PDF in it."
    )

print(f"Loaded {len(chunks)} chunks from PDFs in '{DOCS_FOLDER}/'\n")

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def retrieve(question, top_k=3):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked = sorted(zip(scores, chunk_records), reverse=True, key=lambda x: x[0])
    return [record for score, record in ranked[:top_k]]

# ---- PHASE 2: QUERY + GENERATE ----

def answer(question):
    retrieved = retrieve(question)

    context = "\n".join(f"- (from {r['source']}) {r['text']}" for r in retrieved)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return retrieved, response["message"]["content"]

if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ")
    retrieved, final_answer = answer(question)

    print("\nRetrieved chunks:")
    for r in retrieved:
        print(f"  [{r['source']}] {r['text'][:100]}...")
    print(f"\nAnswer:\n{final_answer}")