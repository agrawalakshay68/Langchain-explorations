from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class Source(BaseModel):
    """Schema for a source of information."""
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""
    answer: str = Field(description="The answer to the user's query.")
    sources: List[Source] = Field(description="A list of sources used to generate the answer.", default_factory=list)

# llm = ChatOllama(model="qwen2.5:3b", temperature=0.2)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

if __name__ == "__main__":
    query = "Search for 3 job postings for an Ai engineer in Bangalore"
    response = agent.invoke({
        "messages": [
            HumanMessage(content=query)
        ]
    })

    print(response)