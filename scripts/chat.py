import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np

load_dotenv()

# ENV variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pmc")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Initialize services
model = SentenceTransformer(MODEL_NAME)
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-pro")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# RAG pipeline: embed query → fetch → prompt Gemini
def chat(query):
    # Step 1: Embed query
    query_embedding = model.encode(query).tolist()

    # Step 2: Query Pinecone
    results = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    # Step 3: Build context from retrieved chunks
    context_chunks = [match["metadata"]["source"] + ":\n" + match["metadata"].get("text", "") for match in results["matches"]]
    context = "\n\n".join(context_chunks)

    if not context:
        return "Sorry, I couldn't find relevant information in the PMC documents."

    # Step 4: Prompt Gemini
    prompt = f"""You are a helpful assistant answering queries from Pune Municipal Corporation (PMC) documents.

Context:
{context}

User question: {query}
Answer:"""

    response = gemini.generate_content(prompt)
    return response.text.strip()

# CLI usage
if __name__ == "__main__":
    while True:
        query = input("\n❓ Ask PMC Bot: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = chat(query)
        print(f"\n🤖 PMC Bot:\n{answer}")
