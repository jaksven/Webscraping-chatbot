import yfinance as yf
from serpapi import GoogleSearch
from langchain import hub
from langchain.agents import Tool, initialize_agent, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import streamlit as st
import warnings
import os
warnings.filterwarnings("ignore")
load_dotenv(".env.local")

google_api_key = os.getenv("GOOGLE_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

#Search duckduckgo for Stock Ticker
def search_ticker(company_name):
    search=DuckDuckGoSearchRun()     
    query = f"{company_name} ticker name"
    result = search.run(query)
    return result

def get_latest_stock_price(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df = df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    return df.iloc[-1]["Close"]

#Get Historical Stock Closing Price for Last 1 Year
def get_stock_price(ticker):
    """
    Analyzes and summarizes the last year of stock price data for a given ticker.
    Returns a concise summary string of key metrics.
    """
    if "." in ticker:
        ticker = ticker.split(".")[0]
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        return "Could not retrieve stock data. The ticker may be invalid."

    # Calculate key metrics
    latest_price = df['Close'].iloc[-1]
    year_high = df['Close'].max()
    year_low = df['Close'].min()
    start_price = df['Close'].iloc[0]
    yearly_change_percent = ((latest_price - start_price) / start_price) * 100
    volatility = df['Close'].std() # Standard deviation as a measure of volatility
    avg_volume = df['Volume'].mean()

    # Create a clean summary string
    summary = (
        f"Stock Price Analysis for {ticker} (Last 1 Year):\n"
        f"- **Latest Closing Price:** ${latest_price:.2f}\n"
        f"- **52-Week High:** ${year_high:.2f}\n"
        f"- **52-Week Low:** ${year_low:.2f}\n"
        f"- **1-Year Price Change:** {yearly_change_percent:.2f}%\n"
        f"- **Volatility (Std. Dev):** ${volatility:.2f}\n"
        f"- **Average Daily Volume:** {avg_volume:,.0f} shares"
    )
    return summary

def google_news(query):
    query += " Stock News"
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": serpapi_key,
        "hl": "en",      # language
        "gl": "us"       # country (geolocation)
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    output = "Recent News:\n"
    for i, news in enumerate(results.get("news_results", [])[:5]):
        title = news.get("title")
        date = news.get("date")
        output += f"News {i+1}:\nTitle of the news article:{title}\nDate of the news article:{date}\n"

    return output

#Get Financial Statements
def get_financial_statements(ticker):
    """
    Retrieves and summarizes the most recent key financial metrics from a company's balance sheet and financials.
    Returns a concise summary string.
    """
    if "." in ticker:
        ticker = ticker.split(".")[0]
    
    company = yf.Ticker(ticker)

    # Use .financials for income statement data and .balancesheet for balance sheet data
    financials = company.financials
    balance_sheet = company.balancesheet
    
    if financials.empty or balance_sheet.empty:
        return "Could not retrieve financial statements. The ticker may be invalid."

    # Get the most recent year's data
    latest_financials = financials.iloc[:, 0]
    latest_balance_sheet = balance_sheet.iloc[:, 0]

    # Extract key metrics, using .get() to avoid errors if a line item is missing
    total_revenue = latest_financials.get('Total Revenue', 'N/A')
    net_income = latest_financials.get('Net Income', 'N/A')
    total_assets = latest_balance_sheet.get('Total Assets', 'N/A')
    total_liabilities = latest_balance_sheet.get('Total Liabilities Net Minority Interest', 'N/A')
    stockholders_equity = latest_balance_sheet.get('Stockholders Equity', 'N/A')

    # Format numbers for readability
    def format_value(value):
        if isinstance(value, (int, float)):
            return f"${value / 1e9:.2f}B" # Format in billions
        return value

    summary = (
        f"Key Financials Summary for {ticker}:\n"
        f"- **Total Revenue:** {format_value(total_revenue)}\n"
        f"- **Net Income:** {format_value(net_income)}\n"
        f"- **Total Assets:** {format_value(total_assets)}\n"
        f"- **Total Liabilities:** {format_value(total_liabilities)}\n"
        f"- **Stockholders' Equity:** {format_value(stockholders_equity)}"
    )
    return summary

def main():

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        max_output_tokens=2048,
        google_api_key=google_api_key 
    )

    #Initialize DuckDuckGo Search Engine
    #search=DuckDuckGoSearchRun()     
    tools = [
    Tool(
        name="Stock Ticker Search",
        func=search_ticker,
        description="Use only when you need to get stock ticker from internet."
    ),
    Tool(
        name = "Get Stock Historical Price",
        func = get_stock_price,
        description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it"
    ),
    Tool(
        name="Get Recent News",
        func= google_news,
        description="Use this to fetch recent news about a specific stock such as 'Apple' or 'Tesla'"
    ),
    Tool(
        name="Get Financial Statements",
        func=get_financial_statements,
        description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it"
    )
    ]

    # 1. Pull a standard, tested ReAct prompt template from the LangChain Hub
    # This prompt is known to work well with ReAct agents.
    prompt_template = hub.pull("hwchase17/react")

    prompt_template.template = """You are an expert financial advisor. Your primary goal is to provide a 'Buy', 'Hold', or 'Sell' recommendation for a given company's stock.

    To achieve this, you must follow a structured analysis using the available tools. You will receive summarized data from these tools.

    **Instructions:**
    1.  First, identify the company's official stock ticker symbol.
    2.  Then, gather essential summarized data: use the tools to get historical stock price analysis, key financial statement metrics, and the latest news.
    3.  Carefully synthesize all the collected summaries. Do not make a decision until all data types (price, financials, news) have been collected.
    4.  Finally, provide a conclusive recommendation justified by the data.

    **You have access to the following tools:**
    {tools}

    **Strictly use the following format:**

    Question: The input question you must answer.
    Thought: Your reasoning on what to do next. This should be concise.
    Action: The action to take, which must be one of [{tool_names}].
    Action Input: The input to the action (e.g., the stock ticker).
    Observation: The summarized result of the action.

    ...(This Thought/Action/Action Input/Observation sequence can repeat up to 10 times)...

    Thought: I have now gathered and analyzed the stock price summary, financial statement summary, and recent news. I have all the necessary information to make a final recommendation.
    Final Answer: Start your response with one word: **Buy**, **Hold**, or **Sell**. Then, provide a detailed, multi-point justification for your decision based on the summarized data you collected.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""


    # 3. Create the agent
    react_agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt_template
    )

    # 4. Create the Agent Executor to run the agent
    # This replaces `initialize_agent`
    zero_shot_agent = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20
    )

    # Your while loop for input remains the same, but the call changes slightly
    while True:
        prompt_input = input("Ask the AI financial advisor: ")
        prompt = f"Should I buy, hold, or sell the stock for the company with the ticker name {prompt_input}?"
        if prompt_input.lower() in ['exit', 'quit']:
            break
        # The input must now be a dictionary
        response = zero_shot_agent.invoke({"input": prompt})
        print(response["output"])
        print("\nClosing Price:", get_latest_stock_price(prompt_input))

if __name__ == "__main__":

    main()

