from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/scholarships.csv"

app = FastAPI(title="🎓 Scholarship Full RAG API")

# -----------------------------
# LOAD MODELS (ONCE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
        print("❌ CSV NOT FOUND")
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
# WEBSITE LOADER (FIXED)
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
    except:
        return []

# -----------------------------
# CREATE VECTOR DB
# -----------------------------
def create_db(docs):
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "csv"
    url: str = ""

# -----------------------------
# CSV ANSWER
# -----------------------------
def generate_csv_answer(results):
    if not results:
        return "No relevant scholarships found."

    answer = f"Found {len(results)} scholarships:\n\n"

    for r in results:
        name = r.metadata.get("name", "")
        category = r.metadata.get("category", "")
        income = r.metadata.get("income", "")

        answer += f"- {name} (Category: {category}, Income: ₹{income})\n"

    answer += "\nCheck links below for details."

    return answer

# -----------------------------
# WEBSITE ANSWER (FINAL CLEAN)
# -----------------------------
def generate_website_answer(results, query):
    if not results:
        return "No information found."

    context = " ".join([r.page_content for r in results])

    sentences = [s.strip() for s in context.split(".")]

    clean_sentences = []
    for s in sentences:
        s_lower = s.lower()

        if len(s) < 60:
            continue

        if any(x in s_lower for x in [
            "see also",
            "not to be confused",
            "references",
            "external links",
            "citation",
            "image",
            "photo",
            "depicts",
            "young man",
            "ceremony"
        ]):
            continue

        clean_sentences.append(s)

    # PRIORITY: definition
    definition_sentences = [
        s for s in clean_sentences if "scholarship" in s.lower()
    ]

    if definition_sentences:
        best = definition_sentences[:2]
    else:
        query_words = query.lower().split()
        ranked = []

        for s in clean_sentences:
            score = sum(word in s.lower() for word in query_words)
            ranked.append((score, s))

        ranked.sort(reverse=True)
        best = [s for _, s in ranked[:2]]

    if not best:
        return context[:200]

    answer = ". ".join(best)

    if not answer.endswith("."):
        answer += "."

    return answer

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/ask")
def ask(req: QueryRequest):

    # -------- CSV MODE --------
    if req.mode == "csv":
        docs = load_csv_docs()

        if not docs:
            return {"error": "CSV not found"}

        db = create_db(docs)
        results = db.similarity_search(req.query, k=5)

        answer = generate_csv_answer(results)

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

    # -------- WEBSITE MODE --------
    elif req.mode == "website":
        if not req.url:
            return {"error": "URL required"}

        docs = load_website(req.url)

        if not docs:
            return {"error": "Failed to load website"}

        db = create_db(docs)
        results = db.similarity_search(req.query, k=3)

        answer = generate_website_answer(results, req.query)

        return {
            "mode": "website",
            "query": req.query,
            "answer": answer,
            "source": req.url
        }

    else:
        return {"error": "Invalid mode"}