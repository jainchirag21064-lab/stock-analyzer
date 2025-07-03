# Indian Stock AI Multi-Agent System with LangChain, OpenAI, IndianAPI, and Streamlit UI

# ---- STEP 0: Setup ----
# Requirements:
# pip install langchain openai pandas requests beautifulsoup4 streamlit googlesearch-python

import os
import requests
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from googlesearch import search
from bs4 import BeautifulSoup
import streamlit as st

# --- Configuration ---
API_KEY = os.getenv("INDIAN_API_KEY")  # Set IndianAPI.in key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Helper: Fetch stock data from Indian API ---
def fetch_stock_data(ticker):
    url = f"https://indianapi.in/indian-stock-market/stock?name={ticker}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    return {
        "ticker": data["tickerId"],
        "price": (data["currentPrice"]["NSE"] or data["currentPrice"]["BSE"]),
        "financials": data["financials"],
        "key_metrics": data["keyMetrics"],
        "analyst_view": data["analystView"],
        "news": " ".join([n["title"] + ": " + n["description"] for n in data["recentNews"]]),
        "sector": data["industry"]
    }

# --- Tool 1: Sentiment Analysis ---
sentiment_template = PromptTemplate(
    input_variables=["text"],
    template="""
    Classify this news/text as Bullish, Neutral, or Bearish. Also summarize major insights in 2 lines:
    
    {text}
    """
)
sentiment_chain = LLMChain(llm=OpenAI(temperature=0), prompt=sentiment_template)

def get_sentiment(news_text):
    return sentiment_chain.run(text=news_text).strip()

# --- Tool 2: Intrinsic Value Estimation ---
def intrinsic_value(financials, metrics, sector):
    eps = financials.get("epsTTM", 0)
    pe = metrics.get("peRatio", 12)
    bv = metrics.get("bookValue", 100)
    if sector in ["IT", "Technology"]:
        return eps * 15
    elif sector in ["Banking"]:
        return bv * 1.3
    else:
        return eps * pe

# --- Tool 3: Web Search for Transcripts/Announcements ---
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

# --- Tool 4: Concall Analysis ---
concall_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
    Based on the following earnings call transcript or management commentary, what are the key growth or risk signals for the stock? Provide a 3-sentence summary:
    
    {transcript}
    """
)
concall_chain = LLMChain(llm=OpenAI(temperature=0), prompt=concall_prompt)

def analyze_concall(ticker):
    query = f"{ticker} latest earnings call transcript site:moneycontrol.com OR site:business-standard.com"
    content = scrape_web_info(query)
    if content:
        return concall_chain.run(transcript=content)
    return "No transcript data found."

# --- Tool 5: Core Analyzer ---
def analyze_stock(ticker):
    try:
        d = fetch_stock_data(ticker)
        sentiment_summary = get_sentiment(d["news"])
        iv = intrinsic_value(d["financials"], d["key_metrics"], d["sector"])
        transcript_summary = analyze_concall(ticker)
        return {
            "ticker": d["ticker"],
            "sector": d["sector"],
            "price": d["price"],
            "intrinsic": round(iv, 2),
            "buy_range": f"{iv*0.9:.2f}-{iv:.2f}",
            "sell_range": f"{iv:.2f}-{iv*1.1:.2f}",
            "sentiment": sentiment_summary,
            "analyst": d["analyst_view"].get("recommendation", "NA"),
            "concall": transcript_summary
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# --- Orchestrator ---
def run_analysis(ticker_list):
    results = [analyze_stock(t.strip()) for t in ticker_list]
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["score"] = (
        (df.intrinsic > df.price).astype(int)
        + df["sentiment"].str.contains("Bullish").astype(int)
    )
    return df.sort_values("score", ascending=False)

# --- Streamlit UI ---
st.set_page_config(page_title="Indian Stock AI Analyzer", layout="wide")
st.title("📊 Indian Stock Multi-Agent AI Analyzer")

input_str = st.text_input("Enter Indian stock tickers (comma-separated, e.g., TCS.NS,INFY.NS)")
if input_str:
    tickers = input_str.split(',')
    with st.spinner("Analyzing..."):
        df = run_analysis(tickers)
        if not df.empty:
            st.success("Analysis Complete")
            st.dataframe(df[["ticker", "sector", "price", "intrinsic", "buy_range", "sell_range", "sentiment", "analyst"]])
            for _, r in df.iterrows():
                st.markdown(f"### {r.ticker}")
                st.write(f"**Sector:** {r.sector}  |  **Price:** ₹{r.price}  |  **IV:** ₹{r.intrinsic}")
                st.write(f"**Buy Range:** {r.buy_range}  |  **Sell Range:** {r.sell_range}")
                st.write(f"**Sentiment & Summary:** {r.sentiment}")
                st.write(f"**Analyst View:** {r.analyst}")
                st.write(f"**Earnings Call Summary:** {r.concall}")
                st.markdown("---")
