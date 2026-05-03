from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI(title="🎓 Scholarship RAG API")

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "csv"   # "csv" or "website"
    url: str | None = None


# -----------------------------
# LOAD CSV (SAFE PATH FOR RENDER)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "scholarships.csv")

try:
    df = pd.read_csv(CSV_PATH)
    df.fillna("", inplace=True)
except Exception as e:
    df = None
    print("❌ CSV LOAD ERROR:", e)


# -----------------------------
# CSV SEARCH FUNCTION
# -----------------------------
def search_scholarships(query: str):
    if df is None:
        return "❌ CSV not loaded", []

    query = query.lower()

    results = df[
        df.apply(lambda row: query in str(row).lower(), axis=1)
    ].head(5)

    if results.empty:
        return "❌ No scholarships found", []

    response = "Found scholarships:\n\n"
    output = []

    for _, row in results.iterrows():
        item = {
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "income": str(row.get("income", "")),
            "apply_link": row.get("apply_link", "")
        }

        response += f"- {item['name']} (Category: {item['category']}, Income: ₹{item['income']})\n"
        output.append(item)

    return response, output


# -----------------------------
# WEBSITE MODE
# -----------------------------
def fetch_website(url: str):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]  # limit size

    except Exception as e:
        return f"❌ Error fetching website: {e}"


# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/ask")
def ask(req: QueryRequest):

    # CSV MODE
    if req.mode == "csv":
        answer, results = search_scholarships(req.query)

        return {
            "mode": "csv",
            "query": req.query,
            "answer": answer,
            "results": results
        }

    # WEBSITE MODE
    elif req.mode == "website":
        if not req.url:
            return {"error": "URL required for website mode"}

        text = fetch_website(req.url)

        return {
            "mode": "website",
            "query": req.query,
            "answer": text,
            "source": req.url
        }

    # INVALID MODE
    return {"error": "Invalid mode"}
