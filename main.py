import yfinance as yf
from serpapi import GoogleSearch
from langchain.agents import Tool, initialize_agent
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

#Get Historical Stock Closing Price for Last 1 Year
def get_stock_price(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df = df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    return df.to_string()

def google_news(query):
    query += " Stock News"
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": serpapi_key,  # replace with your SerpAPI key
        "hl": "en",      # language
        "gl": "us"       # country (geolocation)
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    output = "Recent News:\n"
    for i, news in enumerate(results.get("news_results", [])[:5]):
        title = news.get("title")
        date = news.get("date")
        output += f"News {i+1}:\nTitle of the news article:{title}\nDate of the news article:{date}"

    return output

#Get Financial Statements
def get_financial_statements(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    else:
        ticker=ticker
    company = yf.Ticker(ticker)
    balance_sheet = company.balancesheet
    if balance_sheet.shape[1]>3:
        balance_sheet = balance_sheet.iloc[:,:3]
    balance_sheet = balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

def main():

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=2048,
        google_api_key=google_api_key 
    )

    #Initialize DuckDuckGo Search Engine
    search=DuckDuckGoSearchRun()     
    tools = [
    Tool(
        name="Stock Ticker Search",
        func=search.run,
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

    zero_shot_agent=initialize_agent(
        llm=model,
        agent="zero-shot-react-description",
        tools=tools,
        verbose=True,
        max_iteration=4,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )

    #Adding predefine evaluation steps in the agent Prompt
    stock_prompt="""You are a financial advisor. Give stock recommendations for given query.
    Everytime first you should identify the company name and get the stock ticker symbol for the stock.
    Answer the following questions as best you can. You have access to the following tools:

    Get Stock Historical Price: Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it 
    Stock Ticker Search: Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task
    Get Recent News: Use this to fetch recent news about stocks
    Get Financial Statements: Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it

    steps- 
    Note- if you fail in satisfying any of the step below, Just move to next one
    1) Get the company name and search for the "company name + stock ticker" on internet. Dont hallucinate extract stock ticker as it is from the text. Output- stock ticker. If stock ticker is not found, stop the process and output this text: This stock does not exist
    2) Use "Get Stock Historical Price" tool to gather stock info. Output- Stock data
    3) Get company's historic financial data using "Get Financial Statements". Output- Financial statement
    4) Use this "Get Recent News" tool to search for latest stock related news. Output- Stock news
    5) Analyze the stock based on gathered data and give detailed analysis for investment choice. provide numbers and reasons to justify your answer. Output- Give a single answer if the user should buy,hold or sell. You should Start the answer with Either Buy, Hold, or Sell in Bold after that Justify.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do, Also try to follow steps mentioned above
    Action: the action to take, should be one of [Get Stock Historical Price, Stock Ticker Search, Get Recent News, Get Financial Statements]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times, if Thought is empty go to the next Thought and skip Action/Action Input and Observation)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    zero_shot_agent.agent.llm_chain.prompt.template=stock_prompt

    while True:

        prompt = input("Ask the AI financial advisor: ")

        response = zero_shot_agent(prompt)

        print(response["output"])


if __name__ == "__main__":

    main()

