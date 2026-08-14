from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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

    llm = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0.7)

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

### 1. Short summary
Andrej Karpathy is a Slovak-Canadian AI researcher specializing in deep learning and computer vision. He has worked at OpenAI and Tesla, founded the AI education platform Eureka Labs in 2024, and joined Anthropic’s pretraining team in 2026. He holds degrees from the University of Toronto, the University of British Columbia, and Stanford University.

### 2. Two interesting facts
- As a teenager and young adult, Karpathy ran a YouTube channel called **badmephisto**, where he posted Rubik’s Cube tutorials; the channel had over 9 million views by June 2025.
- His PhD research at Stanford, supervised by Fei-Fei Li, focused on combining **natural language processing, computer vision, and deep learning**.
"""