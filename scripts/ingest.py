import json
import os
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ENV Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pmc")
CLOUD = "gcp"       # or "aws"
REGION = "us-central1"  # or your Pinecone region

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in pc.list_indexes().names():
    print(f"🆕 Creating index '{PINECONE_INDEX}'...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )

index = pc.Index(PINECONE_INDEX)

# Load JSON from Drupal
def load_documents(path="data/raw_data.json"):
    with open(path, "r") as f:
        raw = json.load(f)

    docs = []
    for entry in raw:
        url = entry.get("url")
        records = entry.get("data", {}).get("data", [])
        for item in records:
            if isinstance(item, dict):
                values = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
                text = "\n".join(values).strip()
                if text:
                    docs.append({"text": text, "source": url})
            elif isinstance(item, str):
                docs.append({"text": item.strip(), "source": url})
    return docs

# Split long text into chunks
def split_text(text, max_length=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap
    return chunks

# Embed and upsert to Pinecone
def embed_and_upload():
    print("🔹 Loading documents...")
    raw_docs = load_documents()
    print(f"🗂️ Loaded {len(raw_docs)} documents")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors_to_upsert = []

    print("🔹 Chunking and embedding...")
    for doc in tqdm(raw_docs):
        chunks = split_text(doc["text"])
        embeddings = model.encode(chunks)

        for i, emb in enumerate(embeddings):
            vector_id = str(uuid.uuid4())
            vectors_to_upsert.append({
                "id": vector_id,
                "values": emb.tolist(),
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i
                }
            })

    print(f"⬆️ Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch)

    print("✅ Done uploading all vectors.")

if __name__ == "__main__":
    embed_and_upload()
