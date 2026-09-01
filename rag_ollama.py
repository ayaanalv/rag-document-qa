"""
Full RAG pipeline using Ollama (local, free, no API key).
Run this on YOUR machine, not here -- it needs Ollama running locally.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# ---- PHASE 1: SETUP (same as before) ----

chunks = [
    "The Network Log Analyzer processes CSV datasets to validate and organize network log information.",
    "Binary file I/O and dynamic memory allocation are used to migrate telecom records into binary archives.",
    "Quick Sort and Merge Sort were implemented in C to compare searching and sorting performance.",
    "The Java Arcade Mini-Game Suite uses object-oriented programming to manage game mechanics and scoring.",
]

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def retrieve(question, top_k=2):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [chunk for score, chunk in ranked[:top_k]]

# ---- PHASE 2: QUERY + GENERATE (this part changed) ----

def answer(question):
    retrieved_chunks = retrieve(question)

    context = "\n".join(f"- {c}" for c in retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

    # This talks to the Ollama app running in the background on your machine.
    # No API key, no internet call -- everything happens locally.
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return retrieved_chunks, response["message"]["content"]

if __name__ == "__main__":
    question = "How does the project handle sorting algorithms?"
    retrieved, final_answer = answer(question)

    print(f"Question: {question}\n")
    print("Retrieved chunks (this is what the model was allowed to see):")
    for c in retrieved:
        print(f"  - {c}")
    print(f"\nGenerated answer:\n{final_answer}")