from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import os

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pmc")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Init models
model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-pro")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# FastAPI
app = FastAPI(title="PMC RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Embed query
        query_embedding = model.encode(query).tolist()

        # Step 2: Query Pinecone
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Step 3: Build context
        chunks = [match["metadata"]["source"] + ":\n" + match["metadata"].get("text", "") for match in results["matches"]]
        context = "\n\n".join(chunks)

        if not context:
            return {"answer": "Sorry, I couldn't find any relevant information from PMC."}

        # Step 4: RAG prompt
        prompt = f"""You are an intelligent and helpful assistant for the Pune Municipal Corporation (PMC).You answer user questions clearly, concisely, and naturally — without saying phrases like "Based on the information" or "According to the documents".Use only the relevant information provided below to answer. Be direct, friendly, and informative.

Context:
{context}

User question: {query}
Answer:"""

        response = gemini.generate_content(prompt)
        return {"answer": response.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
