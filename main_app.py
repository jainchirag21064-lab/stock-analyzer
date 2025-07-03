# Indian Stock AI Multi-Agent System with Full Financial Analysis

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

# --- Data Extraction Helpers ---
def extract_eps(financials):
    for fin in financials:
        try:
            for item in fin['stockFinancialMap']['INC']:
                if item['key'] == 'DilutedEPSExcludingExtraOrdItems':
                    return float(item['value'])
        except: continue
    return 0

def extract_metrics_from_keymetrics(key_metrics):
    growth = key_metrics.get("growth", {})
    valuation = key_metrics.get("valuation", {})
    return {
        'peRatio': float(growth.get('priceToEarnings', 12) or 12),
        'bookValue': float(valuation.get('bookValuePerShare', 100) or 100)
    }
    return {'peRatio': 12, 'bookValue': 100}

def extract_news(news_list):
    return " ".join([f"{n['title']}: {n['description']}" for n in news_list])

def extract_shareholding_info(data):
    if not data: return ""
    parts = []
    for item in data:
        parts.append(f"{item['name']}: {item['percentage']}%")
    return ", ".join(parts)

# --- API Integration ---
def fetch_stock_data(ticker):
    url = f"https://indianapi.in/indian-stock-market/stock?name={ticker}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    company_name = data.get("companyName", "")
    eps = extract_eps(data.get("financials", []))
    metrics = extract_metrics_from_keymetrics(data.get("keyMetrics", {}))
    price = float(data.get("currentPrice", {}).get("NSE") or data.get("currentPrice", {}).get("BSE") or 0.0)    
    
    return {
        "ticker": ticker,
        "companyName": company_name,
        "sector": data.get("industry", ""),
        "price": price,
        "eps": eps,
        "pe": metrics['peRatio'],
        "bookValue": metrics['bookValue'],
        "news": extract_news(data.get("recentNews", [])),
        "shareholding": extract_shareholding_info(data.get("shareholding", {}).get("data", [])),
        "technical": data.get("stockTechnicalData", {}),
        "keyMetrics": data.get("keyMetrics", {}),
        "analystView": data.get("analystView", {}),
        "recosBar": data.get("recosBar", {}),
        "details": data.get("stockDetailsReusableData", {}),
        "ratios": data.get("stockFinancialData", {})
    }

# --- Prompt Chains ---
sentiment_prompt = PromptTemplate.from_template("""
Based on the following news, key metrics, and recent developments, classify this stock as Bullish, Neutral or Bearish. Explain the sentiment in 2 lines:

{text}
""")
sentiment_chain = sentiment_prompt | llm

def get_sentiment(text):
    return sentiment_chain.invoke({"text": text}).strip()

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

# --- Valuation ---
def intrinsic_value(eps, pe, book, sector):
    if sector.lower() in ["it", "technology"]:
        return eps * 18  # Higher PE for tech growth
    elif sector.lower() in ["banking", "finance"]:
        return book * 1.5  # Slightly higher than historical P/B
    elif sector.lower() in ["energy", "commodities"]:
        return eps * 8  # Lower PE due to cyclical nature
    else:
        return eps * pe

# --- Core Analyzer ---
def analyze_stock(ticker):
    try:
        d = fetch_stock_data(ticker)
        info_text = f"NEWS: {d['news']}
METRICS: {d['keyMetrics']}
ANALYST VIEW: {d['analystView']}
SHAREHOLDING: {d['shareholding']}
TECHNICAL: {d['technical']}"
        sentiment = get_sentiment(info_text)
        iv = intrinsic_value(d['eps'], d['pe'], d['bookValue'], d['sector'])
        concall_summary = analyze_concall(ticker)

        roe = d['keyMetrics'].get('profitability', {}).get('returnOnEquity', 'NA')
        dy = d['keyMetrics'].get('dividend', {}).get('dividendYield', 'NA')
        de_ratio = d['keyMetrics'].get('solvency', {}).get('debtToEquity', 'NA')

        return {
            "ticker": ticker,
            "sector": d['sector'],
            "price": d['price'],
            "intrinsic": round(iv, 2),
            "buy_range": f"{iv*0.9:.2f}-{iv:.2f}",
            "sell_range": f"{iv:.2f}-{iv*1.1:.2f}",
            "sentiment": sentiment,
            "shareholding": d['shareholding'],
            "concall": concall_summary,
            "pe": d['pe'],
            "bookValue": d['bookValue'],
            "eps": d['eps'],
            "roe": roe,
            "dividendYield": dy,
            "debtEquity": de_ratio
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# --- Aggregator ---
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
                st.write(f"**P/E:** {r['pe']}  |  **Book Value:** {r['bookValue']}  |  **EPS:** {r['eps']}")
                st.write(f"**Sentiment & Summary:** {r['sentiment']}")
                st.write(f"**Shareholding Pattern:** {r['shareholding']}")
                st.write(f"**ROE:** {r['roe']} | **Div Yield:** {r['dividendYield']} | **Debt/Equity:** {r['debtEquity']}")
                st.write(f"**Earnings Call Summary:** {r['concall']}")
                st.markdown("---")
