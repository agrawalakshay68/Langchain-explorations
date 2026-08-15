from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

@tool
def search_tool(query: str) -> str:
    """
    A simple search tool that takes a query string and returns a search result.

    arguments:
    - query: The search query string.

    returns:
    - A string representing the search result.
    """
    # For demonstration purposes, we'll just return a mock search result.
    # In a real implementation, you would integrate with a search engine or API.
    print(f"Searching for: {query}")
    return f"New Delhi"

llm = ChatOllama(model="qwen2.5:3b")
tools = [search_tool]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    query = "What is the capital of India?"
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
    })

    print(result)

if __name__ == "__main__":
    main()

"""
Code Output:

Searching for: capital of India

{'messages': [HumanMessage(content='What is the capital of India?', additional_kwargs={}, response_metadata={}, id='69859f7c-8e98-4d50-ade4-923bfdf2bf0b'), AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'qwen2.5:3b', 'created_at': '2026-08-15T17:28:12.3836658Z', 'done': True, 'done_reason': 'stop', 'total_duration': 7377521900, 'load_duration': 6029143300, 'prompt_eval_count': 180, 'prompt_eval_duration': 876466000, 'eval_count': 22, 'eval_duration': 462505000, 'logprobs': None, 'model_name': 'qwen2.5:3b', 'model_provider': 'ollama'}, id='lc_run--01a00677-6c8a-7173-b512-cedd450b4849-0', tool_calls=[{'name': 'search_tool', 'args': {'query': 'capital of India'}, 'id': '4dbe7a23-3e91-4eaa-b659-960ac2f9f778', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 180, 'output_tokens': 22, 'total_tokens': 202}), ToolMessage(content='New Delhi', name='search_tool', id='2dd380de-ccc5-4363-b1fd-96099e34f55b', tool_call_id='4dbe7a23-3e91-4eaa-b659-960ac2f9f778'), AIMessage(content='The capital of India is New Delhi.', additional_kwargs={}, response_metadata={'model': 'qwen2.5:3b', 'created_at': '2026-08-15T17:28:13.0126847Z', 'done': True, 'done_reason': 'stop', 'total_duration': 616554500, 'load_duration': 296363800, 'prompt_eval_count': 221, 'prompt_eval_duration': 142320000, 'eval_count': 9, 'eval_duration': 170271000, 'logprobs': None, 'model_name': 'qwen2.5:3b', 'model_provider': 'ollama'}, id='lc_run--01a00677-8969-72d0-b2a1-1021e3562623-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 221, 'output_tokens': 9, 'total_tokens': 230})]}

"""
