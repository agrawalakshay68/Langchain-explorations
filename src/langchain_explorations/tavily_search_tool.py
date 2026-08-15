from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search_tool(query: str):
    """
    A simple search tool that takes a query string and returns a search result.

    arguments:
    - query: The search query string.

    returns:
    - A string representing the search result.
    """

    print(f"Searching for: {query}")
    return tavily.search(query = query)

llm = ChatOpenAI(model="gpt-5")

tools = [search_tool]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    query = query = "Search the job openings for Software Engineer/AI Engineer in India for 3+ years of experience and list their details and apply links."
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
    })

    print(result)

if __name__ == "__main__":
    main()

"""
Code Output:
Searching for: site:linkedin.com/jobs "AI Engineer" India "3+ years"
Searching for: site:linkedin.com/jobs "Software Engineer" AI India "3+ years"
Searching for: site:naukri.com "AI Engineer" 3-6 yrs India
Searching for: site:naukri.com "Machine Learning Engineer" 3-6 yrs India
Searching for: site:indeed.co.in "AI Engineer" India "3 years"
Searching for: site:indeed.co.in "Machine Learning Engineer" India 3 years
Searching for: site:careers.google.com India "Machine Learning Engineer"
Searching for: site:careers.microsoft.com "Software Engineer II" India
Searching for: site:amazon.jobs "Machine Learning Engineer" India
Searching for: site:careers.flipkart.com "Software Development Engineer II"
Searching for: site:jobs.nvidia.com India "Deep Learning"
Searching for: site:careers.phonepe.com "Machine Learning" India
Searching for: site:razorpay.com/careers "Machine Learning"
Searching for: site:careers.swiggy.com "Machine Learning"
Searching for: site:careers.zoho.com "Machine Learning" India
Searching for: site:careers.freshworks.com "Machine Learning" India
Searching for: site:careers.flipkart.com "Machine Learning Engineer"
Searching for: site:linkedin.com/jobs "Machine Learning Engineer" India "3-6 years"
Searching for: site:naukri.com "LLM Engineer" 3-6 Yrs India
Searching for: site:careers.phonepe.com "Machine Learning Engineer"
Searching for: site:boards.greenhouse.io India "Machine Learning Engineer"
Searching for: site:jobs.lever.co India "Machine Learning Engineer"
Searching for: "LLM Engineer" India "3+ years"
Searching for: "AI Engineer" India "3+ years" site:jobs.ashbyhq.com
Searching for: site:linkedin.com/jobs "Generative AI Engineer" India "3-5 years"
Searching for: site:naukri.com "Generative AI Engineer" 3-5 Yrs India
Searching for: site:careers.flipkart.com "Machine Learning Engineer"

Here are current Software Engineer / AI-Engineer openings in India that call for 3+ years’ experience. I’ve included quick details and the direct apply/view links.
Lead AI Engineer — White Space Global — Rajkot, Gujarat
Experience: 3–6 years
Note: Lead development of advanced AI system (CrewAI/agentic).
Apply: https://in.linkedin.com/jobs/view/lead-ai-engineer-rajkot-3-to-6-years-of-work-experience-only-apply-at-white-space-global%C2%AE-4192366642
Artificial Intelligence Engineer — Tomorrow World Technologies (TWT) — India
Experience: 3+ years ML; 3+ years Python/DSA
Apply: https://in.linkedin.com/jobs/view/artificial-intelligence-engineer-at-tomorrow-world-technologies-twt-4323488620
Software Engineer — Simbian AI — India
Experience: 3+ years (production-grade software)
Apply: https://in.linkedin.com/jobs/view/software-engineer-at-simbian-ai-3766478332
AI/ML Engineer — Jobgether — India
Experience: 3–6 years (hands-on ML engineering)
Apply: https://in.linkedin.com/jobs/view/ai-ml-engineer-at-jobgether-4452786238
Machine Learning Engineer — Talentgigs — Coimbatore (WFO)
Experience: 3–6 years
Apply: https://in.linkedin.com/jobs/view/machine-learning-engineer-at-talentgigs-4453905505
Machine Learning Engineer (ML/DL/GenAI), Mid-Level — Capestart — India
Experience: 3–6 years
Apply: https://in.linkedin.com/jobs/view/machine-learning-engineer-mid-level-at-capestart-4416630757
Machine Learning–AI and Data Science Engineer II — Deloitte — India
Experience: 3–6 years (ML/DS/statistical modeling)
Apply: https://in.linkedin.com/jobs/view/machine-learning-ai-and-data-science-engineer-ii-at-deloitte-4452989778
AI Engineer — Marsh McLennan Global Services India — Gurugram
Experience: 3–6 years
Apply: https://www.naukri.com/job-listings-ai-engineer-marsh-mclennan-global-services-india-private-limited-gurugram-3-to-6-years-010426914294
AI Engineer — Elowit Technologies — Bengaluru
Experience: 3–6 years (LLMs, MLOps, Docker/K8s, RAG etc. per listing)
Apply: https://www.naukri.com/job-listings-ai-engineer-elowit-technolgies-pvt-ltd-bengaluru-3-to-6-years-060826927407
Senior Machine Learning Engineer — UPS Supply Chain Solutions — Chennai
Experience: 3–9 years (model support/monitoring in production)
Apply: https://www.naukri.com/job-listings-senior-machine-learning-engineer-ups-pvt-ltd-chennai-3-to-9-years-210224501576
ML Engineer III — 6sense — India
Experience: Mid/senior IC (ML lifecycle; apply page via Greenhouse)
Apply: https://boards.greenhouse.io/6sense/jobs/8106123
AI Engineer II / III — KlearNow.ai — Gurgaon (On-site)
Experience: Mid–senior (AI/ML platform for trade docs)
Apply: https://jobs.lever.co/klearnow/e95fe4bc-b0a0-44cd-915b-2cd9ac578e71
System Software Engineer – Deep Learning — NVIDIA — Bengaluru
Experience: 5+ years (DL systems)
Apply: http://jobs.nvidia.com/careers/job/893392889082
AI Developer II — OpenGov — Pune, India
Experience: Mid-level AI developer (GenAI/AI apps; services)
Apply: https://jobs.ashbyhq.com/opengov/ea5f3b39-867a-4dd4-8585-9a3dc6638cf5

Notes:
Job availability changes quickly; if a link shows “no longer accepting,” tell me and I’ll replace it with similar active roles.
If you share your city preferences (e.g., Bengaluru, Pune, Hyderabad, NCR), domain focus (GenAI/LLM, CV, NLP, MLOps, backend-for-AI), and work mode (remote/hybrid/on-site), I can narrow this to the best-matching openings and draft tailored application notes.
"""
