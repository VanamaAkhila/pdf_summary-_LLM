!pip install faiss-cpu sentence-transformers transformers PyPDF2
from google.colab import files
uploaded = files.upload()   # Choose your research paper PDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import textwrap
def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200):
    text = text.replace("\r", " ").replace("\n", " ")
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j, length = i, 0
        while j < len(words) and length + len(words[j]) + 1 <= chunk_size:
            length += len(words[j]) + 1
            j += 1
        chunk = " ".join(words[i:j])
        chunks.append(chunk.strip())
        i = j - overlap // 5  # overlap approx
        if i < 0: i = 0
        if j == len(words): break
    return chunks
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32"), embedder
def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index
def retrieve(index, embeddings, query_embedding, top_k=5):
    q = np.asarray([query_embedding]).astype("float32")
    D, I = index.search(q, top_k)
    return [(int(idx), float(score)) for idx, score in zip(I[0], D[0])]
def load_summarizer(model_name="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return summarizer
def make_prompt_for_summary(retrieved_texts, instruction="Summarize this research paper in 4-6 sentences focusing on contributions and results:"):
    return f"{instruction}\n\nContext:\n{'\n\n'.join(retrieved_texts)}\n\nSummary:"
def summarize(summarizer, prompt, max_length=200):
    out = summarizer(prompt, max_length=max_length, do_sample=False)
    return out[0]['generated_text']
pdf_path = "Line_Follower_Robot_with_Obstacle_Avoiding_Module.pdf"
text = load_pdf_text(pdf_path)
print("Total characters:", len(text))chunks = chunk_text(text)
print("Chunks:", len(chunks))
embeddings, embedder = embed_chunks(chunks)
index = build_faiss_index(embeddings)
query = "Summarize the paper focusing on main contributions and results."
q_emb = embedder.encode([query], convert_to_numpy=True, 
normalize_embeddings=True)[0].astype("float32")
results = retrieve(index, embeddings, q_emb, top_k=6)
retrieved_texts = [chunks[idx] for idx, _ in results]
prompt = make_prompt_for_summary(retrieved_texts)
summarizer = load_summarizer()
summary = summarize(summarizer, prompt)
print("\n===== AI Generated Summary =====\n")
print(summary)

