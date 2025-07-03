# Indian Stock AI Multi-Agent System with LangChain, OpenAI, IndianAPI, and Streamlit UI

# ---- STEP 0: Setup ----
# Requirements:
# pip install langchain openai pandas requests beautifulsoup4 streamlit googlesearch-python

import os
import requests
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from googlesearch import search
from bs4 import BeautifulSoup
import streamlit as st

# --- Configuration ---
API_KEY = os.getenv("INDIAN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = OpenAI(temperature=0)

# --- Helper: Fetch stock data ---
def extract_eps_from_financials(financials):
    for fin in financials:
        try:
            for item in fin['stockFinancialMap']['INC']:
                if item['key'] == 'DilutedEPSExcludingExtraOrdItems':
                    return float(item['value'])
        except: continue
    return None

def extract_metrics(peer_list, company_name):
    for peer in peer_list:
        if peer['companyName'].lower() in company_name.lower():
            return {
                'peRatio': float(peer.get('priceToEarningsValueRatio', 0) or 0),
                'bookValue': float(peer.get('priceToBookValueRatio', 0) or 0)
            }
    return {'peRatio': 12, 'bookValue': 100}

def fetch_stock_data(ticker):
    url = f"https://indianapi.in/indian-stock-market/stock?name={ticker}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    
    eps = extract_eps_from_financials(data.get("financials", []))
    metrics = extract_metrics(data.get("peerCompanyList", []), data.get("companyName", ""))
    price = float(data.get("currentPrice", {}).get("NSE") or data.get("currentPrice", {}).get("BSE") or 0)

    return {
        "ticker": ticker,
        "companyName": data.get("companyName", ""),
        "sector": data.get("industry", ""),
        "price": price,
        "eps": eps or 0,
        "pe": metrics['peRatio'],
        "bookValue": metrics['bookValue'],
        "news": data.get("companyProfile", {}).get("companyDescription", "")
    }

# --- Tool: Sentiment Chain ---
sentiment_prompt = PromptTemplate.from_template("""
Classify this news/text as Bullish, Neutral, or Bearish. Also summarize major insights in 2 lines:

{text}
""")
sentiment_chain = sentiment_prompt | llm

def get_sentiment(news_text):
    return sentiment_chain.invoke({"text": news_text}).strip()

# --- Tool: Web Transcript Search ---
def scrape_web_info(query):
    links = list(search(query, num_results=3))
    summaries = []
    for url in links:
        try:
            html = requests.get(url, timeout=5).text
            soup = BeautifulSoup(html, 'html.parser')
            text = ' '.join([p.text for p in soup.find_all('p')])
            summaries.append(text[:2000])
        except:
            continue
    return " ".join(summaries)

concall_prompt = PromptTemplate.from_template("""
Based on the following earnings call transcript or management commentary, what are the key growth or risk signals for the stock? Provide a 3-sentence summary:

{transcript}
""")
concall_chain = concall_prompt | llm

def analyze_concall(ticker):
    query = f"{ticker} latest earnings call transcript site:moneycontrol.com OR site:business-standard.com"
    content = scrape_web_info(query)
    if content:
        return concall_chain.invoke({"transcript": content}).strip()
    return "No transcript data found."

# --- Tool: Valuation ---
def intrinsic_value(eps, pe, book, sector):
    if sector.lower() in ["it", "technology"]:
        return eps * 15
    elif sector.lower() in ["banking", "finance"]:
        return book * 1.3
    else:
        return eps * pe

# --- Stock Analyzer ---
def analyze_stock(ticker):
    try:
        d = fetch_stock_data(ticker)
        sentiment = get_sentiment(d["news"])
        iv = intrinsic_value(d["eps"], d["pe"], d["bookValue"], d["sector"])
        concall_summary = analyze_concall(ticker)

        return {
            "ticker": ticker,
            "sector": d["sector"],
            "price": d["price"],
            "intrinsic": round(iv, 2),
            "buy_range": f"{iv*0.9:.2f}-{iv:.2f}",
            "sell_range": f"{iv:.2f}-{iv*1.1:.2f}",
            "sentiment": sentiment,
            "concall": concall_summary
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# --- Orchestrator ---
def run_analysis(tickers):
    results = [analyze_stock(t.strip()) for t in tickers]
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["score"] = ((df["intrinsic"] > df["price"]).astype(int) + df["sentiment"].str.contains("Bullish").astype(int))
    return df.sort_values("score", ascending=False)

# --- Streamlit UI ---
st.set_page_config(page_title="Indian Stock AI Analyzer", layout="wide")
st.title("📊 Indian Stock Multi-Agent AI Analyzer")

input_str = st.text_input("Enter Indian stock tickers (comma-separated, e.g., TATASTEEL)")
if input_str:
    tickers = input_str.split(',')
    with st.spinner("Analyzing..."):
        df = run_analysis(tickers)
        if not df.empty:
            st.success("Analysis Complete")
            st.dataframe(df[["ticker", "sector", "price", "intrinsic", "buy_range", "sell_range", "sentiment"]])
            for _, r in df.iterrows():
                st.markdown(f"### {r['ticker']}")
                st.write(f"**Sector:** {r['sector']}  |  **Price:** ₹{r['price']}  |  **IV:** ₹{r['intrinsic']}")
                st.write(f"**Buy Range:** {r['buy_range']}  |  **Sell Range:** {r['sell_range']}")
                st.write(f"**Sentiment & Summary:** {r['sentiment']}")
                st.write(f"**Earnings Call Summary:** {r['concall']}")
                st.markdown("---")
