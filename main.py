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

    # 1. Pull a standard, tested ReAct prompt template from the LangChain Hub
    # This prompt is known to work well with ReAct agents.
    prompt_template = hub.pull("hwchase17/react")

    # 2. Define your high-level instructions within the prompt's variables
    prompt_template.template = """You are an expert financial advisor. Your goal is to provide a 'Buy', 'Hold', or 'Sell' recommendation for a given company's stock.

    To do this, you will perform a structured analysis by using the available tools to gather all necessary information before answering.
    1.  First, identify the company and find its official stock ticker symbol.
    2.  Then, gather essential data: historical stock prices, recent financial statements, and the latest news.
    3.  Synthesize all the collected information.
    4.  Finally, provide a conclusive recommendation justified by the data.

    You have access to the following tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I have gathered all necessary data and am ready to provide the final analysis.
    Final Answer: Start your response with **Buy**, **Hold**, or **Sell**. Then, provide a concise but detailed justification.

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
        max_iterations=6
    )

    # Your while loop for input remains the same, but the call changes slightly
    while True:
        prompt_input = input("Ask the AI financial advisor: ")
        if prompt_input.lower() in ['exit', 'quit']:
            break
        # The input must now be a dictionary
        response = zero_shot_agent.invoke({"input": prompt_input})
        print(response["output"])


if __name__ == "__main__":

    main()

