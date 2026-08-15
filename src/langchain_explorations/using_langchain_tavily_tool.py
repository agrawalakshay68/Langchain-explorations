from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(model="gpt-5")

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    query = "Search for jobs of Ai engineer with 3+ years of experience in India?"
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
    })

    print(result)

if __name__ == "__main__":
    main()

"""
Code Output:

Here are current AI/ML/GenAI engineer openings in India that fit roughly 3+ years of experience (found within the last month). Some require sign‑in to view/apply.
Unicorn Workforce — AI Engineer (2–3 yrs, Remote, 6‑month contract). LinkedIn: https://in.linkedin.com/jobs/view/position-ai-engineer-experience-2%E2%80%933-years-work-mode-remote-contract-duration-6-months-at-unicorn-workforce-4453669563
Hewlett Packard Enterprise — Generative AI Engineer (2–4 yrs, Bengaluru). Naukri: https://www.naukri.com/generative-ai-jobs-4
Impetus Technologies — Generative AI Engineer (4–9 yrs, Gurugram/Chennai/Bengaluru). Naukri: https://www.naukri.com/generative-ai-jobs-4
Saint Gobain — Generative AI Engineer (4–7 yrs, Mumbai/Navi Mumbai). Naukri: https://www.naukri.com/generative-ai-jobs-4
Overture Rede — Chatbot Developer (Conversational AI) (3–6 yrs, Mumbai). Naukri: https://www.naukri.com/conversational-ai-jobs
Advaitha Information Technology Services — AI/NLP/Conversational AI Engineer (3–4 yrs, Hyderabad). Naukri: https://www.naukri.com/conversational-ai-jobs
Xminds Infotech — Conversational AI Engineer (4+ yrs, Thiruvananthapuram). Naukri: https://www.naukri.com/conversational-ai-jobs
First Advantage — Senior AI Automation Engineer (Powershell, AI Solutions & Chatbots) (3–7 yrs, Bengaluru). Naukri (AI Automation): https://www.naukri.com/ai-automation-jobs
HSBC — Machine Learning Data Scientist (3–7 yrs, Bengaluru). Naukri (Python/ML): https://www.naukri.com/python-machine-learning-jobs
EY — Data Scientist (Machine Learning, GenAI) (4–9 yrs, Hybrid India). Naukri (Python/ML): https://www.naukri.com/python-machine-learning-jobs
Amgen — Data Engineer, Translational Data Management, Automation & AI (3+ yrs data eng; overall 8+ total exp; Hyderabad). Amgen Careers: https://careers.amgen.com/en/job/hyderabad/data-engineer-translational-data-management-automation-and-ai/87/97228929120

More listings and active boards to browse right now:
LinkedIn (AI Engineer, India; includes 2–3 and 3–5 yr roles): https://in.linkedin.com/jobs/ (example listing above + similar roles on the page)
Indeed India — Machine Learning/AI Engineer: https://www.indeed.co.in/Machine-Learning-Engineer-jobs
Naukri — Generative AI jobs: https://www.naukri.com/generative-ai-jobs-4
Naukri — Conversational AI jobs: https://www.naukri.com/conversational-ai-jobs
Naukri — AI Automation jobs: https://www.naukri.com/ai-automation-jobs

Want me to narrow this to roles that best match you? Tell me:
Preferred cities (e.g., Bengaluru, Hyderabad, Pune, NCR) or remote/hybrid
Focus area (GenAI/LLMs, NLP, CV, MLOps, platform engineering)
Target salary range and company types (product, startup, MNC) I can also set up alerts and tailor your resume/cover note for these roles.

"""
