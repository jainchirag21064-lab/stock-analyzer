# main.py
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import openai
import newspaper
from newspaper import Article

openai.api_key = "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"

st.set_page_config(page_title="Agentic AI Stock Analyzer", layout="wide")
st.title("📈 Agentic AI Stock Analyzer")

# --- Input ---
ticker = st.text_input("Enter Stock Ticker (e.g. TCS.NS)")

# --- Helper Functions ---
def get_news(ticker_name):
    url = f"https://news.google.com/search?q={ticker_name}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = ["https://news.google.com" + a['href'][1:] for a in soup.find_all('a', href=True) if '/articles/' in a['href']]
    return links[:3]

def summarize_and_sentiment(article_url):
    article = Article(article_url)
    article.download()
    article.parse()
    article.nlp()

    summary = article.summary

    sentiment_prompt = f"""
    Analyze the following news and give sentiment as Positive, Negative or Neutral for the company mentioned.
    News: {summary}
    """

    sentiment = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": sentiment_prompt}]
    )["choices"][0]["message"]["content"]

    return summary, sentiment


def get_intrinsic_value(info, sector):
    prompt = f"""
    You are a finance expert. Based on the following financials, estimate the intrinsic value for a company in the {sector} sector.

    PE Ratio: {info.get('trailingPE')}
    EPS: {info.get('trailingEps')}
    Market Cap: {info.get('marketCap')}
    Book Value: {info.get('bookValue')}

    Use appropriate valuation method (e.g., DCF, PE multiple, etc) for the sector.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# --- Main Logic ---
if ticker:
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get("sector", "Unknown")

    st.subheader("📌 Company Overview")
    st.write(f"**Company:** {info.get('longName')}")
    st.write(f"**Sector:** {sector}")
    st.write(f"**Market Cap:** {info.get('marketCap')}")
    st.write(f"**PE Ratio:** {info.get('trailingPE')}")

    st.subheader("📊 Financial Statements")
    try:
        st.write(stock.financials)
        st.write(stock.quarterly_financials)
    except:
        st.warning("Could not load financials.")

    st.subheader("🧠 AI-based Intrinsic Value")
    try:
        intrinsic_value = get_intrinsic_value(info, sector)
        st.success("Estimated Intrinsic Value:")
        st.write(intrinsic_value)
    except:
        st.warning("Failed to estimate intrinsic value.")

    st.subheader("📈 Analyst Recommendations")
    try:
        st.write(stock.recommendations.tail(5))
    except:
        st.warning("No analyst recommendation data found.")

    st.subheader("📰 News & Sentiment Analysis")
    try:
        news_links = get_news(info.get("longName", ticker))
        for link in news_links:
            try:
                summary, sentiment = summarize_and_sentiment(link)
                st.markdown(f"**[News Article]({link})**")
                st.write(f"**Summary:** {summary}")
                st.write(f"**Sentiment:** {sentiment}")
            except:
                st.write("Could not process news article.")
    except:
        st.warning("Failed to fetch news.")
