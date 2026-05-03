from fastapi import FastAPI
import pandas as pd
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# Load CSV once
try:
    df = pd.read_csv("data/scholarships.csv")
except:
    df = None


@app.get("/")
def home():
    return {"message": "Scholarship API is running 🚀"}


@app.get("/scholarships")
def get_scholarships():
    if df is None:
        return {"error": "CSV not loaded"}
    return df.head(10).to_dict(orient="records")


@app.get("/search")
def search_scholarships(query: str):
    if df is None:
        return {"error": "CSV not loaded"}

    results = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
    return results.head(10).to_dict(orient="records")


@app.get("/scrape")
def scrape_scholarships(url: str):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        links = [a.text for a in soup.find_all("a")[:20]]

        return {"data": links}
    except Exception as e:
        return {"error": str(e)}
