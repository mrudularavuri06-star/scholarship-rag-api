from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/scholarships.csv"

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="🎓 Scholarship LLM RAG API")

@app.get("/")
def home():
    return {"message": "LLM-powered Scholarship RAG API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------
# GLOBAL CACHE
# -----------------------------
embeddings = None
db_cache = None
llm_tokenizer = None
llm_model = None

# -----------------------------
# LOAD EMBEDDINGS (LAZY)
# -----------------------------
def get_embeddings():
    global embeddings
    if embeddings is None:
        logging.info("🔄 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

# -----------------------------
# LOAD LLM (LAZY)
# -----------------------------
def get_llm():
    global llm_tokenizer, llm_model

    if llm_model is None:
        logging.info("🔄 Loading LLM model...")
        model_name = "google/flan-t5-small"

        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return llm_tokenizer, llm_model

# -----------------------------
# TEXT SPLITTER
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

# -----------------------------
# LOAD CSV DATA
# -----------------------------
def load_csv_docs():
    docs = []

    if not os.path.exists(DATA_PATH):
        logging.error(f"❌ CSV NOT FOUND at {DATA_PATH}")
        return docs

    df = pd.read_csv(DATA_PATH).fillna("")

    for _, row in df.iterrows():
        text = f"""
Scholarship Name: {row.get('Name','')}
Category: {row.get('Category','')}
Income Limit: {row.get('Income_Limit','')}
Minimum Marks: {row.get('Min_mark','')}
Benefits: {row.get('Benefits','')}
Deadline: {row.get('End_date','')}
Apply Link: {row.get('Apply_link','')}
Description: {row.get('Description','')}
"""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": row.get("Name", ""),
                    "category": row.get("Category", ""),
                    "income": row.get("Income_Limit", ""),
                    "link": row.get("Apply_link", "")
                }
            )
        )

    return docs

# -----------------------------
# WEBSITE LOADER
# -----------------------------
def load_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = " ".join(soup.get_text().split())

        return [
            Document(
                page_content=text[:4000],
                metadata={"source": url}
            )
        ]

    except Exception as e:
        logging.error(f"Website load error: {e}")
        return []

# -----------------------------
# CREATE VECTOR DB
# -----------------------------
def create_db(docs):
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())

# -----------------------------
# CACHE CSV DB (LAZY)
# -----------------------------
def get_csv_db():
    global db_cache

    if db_cache is not None:
        return db_cache

    logging.info("🔄 Creating FAISS DB from CSV...")
    docs = load_csv_docs()

    if not docs:
        return None

    db_cache = create_db(docs)
    return db_cache

# -----------------------------
# LLM ANSWER GENERATION
# -----------------------------
def generate_llm_answer(query, docs):
    if not docs:
        return "No relevant information found."

    tokenizer, model = get_llm()

    context = "\n\n".join([d.page_content for d in docs[:5]])

    prompt = f"""
You are a helpful scholarship assistant.

Instructions:
- Answer ONLY from the context
- If not found, say "I don't know"
- Keep answers clear and structured
- Mention scholarship names if available

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "csv"
    url: str = ""

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/ask")
def ask(req: QueryRequest):

    try:
        if req.mode == "csv":
            db = get_csv_db()

            if db is None:
                return {"error": "CSV not found"}

            results = db.similarity_search(req.query, k=5)
            answer = generate_llm_answer(req.query, results)

            structured = [
                {
                    "name": r.metadata.get("name"),
                    "category": r.metadata.get("category"),
                    "income": r.metadata.get("income"),
                    "apply_link": r.metadata.get("link")
                }
                for r in results
            ]

            return {
                "mode": "csv",
                "query": req.query,
                "answer": answer,
                "results": structured
            }

        elif req.mode == "website":
            if not req.url:
                return {"error": "URL required"}

            docs = load_website(req.url)

            if not docs:
                return {"error": "Failed to load website"}

            db = create_db(docs)
            results = db.similarity_search(req.query, k=3)

            answer = generate_llm_answer(req.query, results)

            return {
                "mode": "website",
                "query": req.query,
                "answer": answer,
                "source": req.url
            }

        else:
            return {"error": "Invalid mode"}

    except Exception as e:
        logging.error(f"❌ API Error: {e}")
        return {"error": "Something went wrong"}