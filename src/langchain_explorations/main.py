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

llm = ChatOpenAI()
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

{'messages': [HumanMessage(content='What is the capital of India?', additional_kwargs={}, response_metadata={}, id='011b4047-6bb7-4c56-a3b3-ed3c8135561c'), AIMessage(content='', additional_kwargs={'refusal': None
        }, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 86, 'total_tokens': 102, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None
                }, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None
                }
            }, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-EDCNXNy2UiQR93LSUZ2DJGIJ1Qdxw', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None
        }, id='lc_run--01a00672-6176-7903-9241-d019b0907960-0', tool_calls=[
            {'name': 'search_tool', 'args': {'query': 'capital of India'
                },'id': 'call_7sez6Y0a66XWtHbV0IvMDY0B', 'type': 'tool_call'
            }
        ], invalid_tool_calls=[], usage_metadata={'input_tokens': 86, 'output_tokens': 16, 'total_tokens': 102, 'input_token_details': {'audio': 0, 'cache_read': 0
            }, 'output_token_details': {'audio': 0, 'reasoning': 0
            }
        }), ToolMessage(content='The capital of India is New Delhi.', name='search_tool', id='6817a4a6-aefb-4a88-88eb-2f7c648ec88c', tool_call_id='call_7sez6Y0a66XWtHbV0IvMDY0B'), AIMessage(content='The capital of India is New Delhi.', additional_kwargs={'refusal': None
        }, response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 118, 'total_tokens': 127, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None
                }, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None
                }
            }, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-EDCNaLaiQUrTHF2p4U7tCBcAbqaGn', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None
        }, id='lc_run--01a00672-6d0a-79c2-9cb0-6a4e0fc3d89d-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 118, 'output_tokens': 9, 'total_tokens': 127, 'input_token_details': {'audio': 0, 'cache_read': 0
            }, 'output_token_details': {'audio': 0, 'reasoning': 0
            }
        })
    ]
}
"""
