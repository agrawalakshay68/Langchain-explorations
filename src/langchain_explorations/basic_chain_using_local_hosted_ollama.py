from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    information = """"
            Andrej Karpathy (born 23 October 1986[3]) is a Slovak-Canadian AI researcher,[4] who co-founded and formerly worked at OpenAI,[5][6][7] where he specialized in deep learning and computer vision.[8][9][2] He also worked as the director of artificial intelligence and Autopilot Vision at Tesla, and in 2024 he founded Eureka Labs, an AI education platform.[10] In 2026 he joined Anthropic as part of the pretraining team.[11]
            Education and early life Karpathy was born in Bratislava, Czechoslovakia (now Slovakia),[12][13][14][15] and moved with his family to Toronto when he was 15.[4] He completed his Computer Science and Physics bachelor's degrees at the University of Toronto in 2009[16] and his master's degree at the University of British Columbia in 2011,[16] where he worked on physically simulated figures.
            In 2006, Karpathy began posting videos on YouTube on his channel, badmephisto. He garnered fame by posting Rubik's cube tutorials which have been used by famous speedcubers such as Feliks Zemdegs.[17] The channel has over 9 million views as of June 2025.
            Karpathy received a PhD from Stanford University in 2015 under the supervision of Fei-Fei Li, focusing on the intersection of natural language processing and computer vision, and deep learning models suited for this task.[18][19][20]"""

    summary_template = PromptTemplate(
        input_variables=["information"],
        template="""
            Given the following information about a person, provide
            1. A short summary
            2. two interesting facts about the person
            Information: {information}
        """
    )

    llm = ChatOllama(model="qwen2.5:3b", temperature=0.7)

    # using LCEL
    chain = summary_template | llm

    response = chain.invoke({
        "information": information
    })

    print(response.content)

if __name__ == "__main__":
    main()

"""
Model output:

Okay, here’s a summary, two interesting facts, and an analysis of the provided information about Andrej Karpathy:

**1. Short Summary:**

Adej Karpathy is a leading AI researcher and entrepreneur specializing in deep learning, computer vision, and automated machine learning. He co-founded and worked at OpenAI and Eureka Labs, and he recently joined Anthropic as part of a pretraining team. His background is rooted in computer science and physics, with a significant focus on creating algorithms that can learn and perform complex tasks through simulated examples.

**2. Two Interesting Facts:**

*   **YouTube Fame:** His channel, "badmephisto," currently boasts over 9 million views and has grown into a significant platform for sharing Rubik's cube tutorials – a testament to his innovative approach to visual learning and problem-solving.
*   **Specific Research Focus:** He initially focused his research on the intersection of NLP and computer vision, a crucial area that informed his later work on large-scale AI models and deep learning techniques.


**Here's a little deeper analysis based on the text:**

*   **Early Career & Foundation:** Karpathy’s early career was shaped by a move to Toronto and a strong foundation in computer science and physics from the University of Toronto.
*   **Deep Learning Pioneer:** His doctoral work with Professor Fei-Fei Li centered around developing deep learning techniques specifically for natural language processing and computer vision – a very ambitious and original area of research. 
*  **Recent Transition:** His recent move to Anthropic and his involvement in the pretraining team indicate a growing focus on practical AI applications and models.

Would you like me to elaborate on any of these points or explore a specific aspect of his career or background further?
"""