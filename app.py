import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0,
    max_retries=0,
)

print("Gemini connection setup successful!")

#cell 2
from typing import TypedDict, List, Optional


class PlacementState(TypedDict):
    resume: str
    job_description: str

    required_skills: List[str]
    candidate_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]

    improvement_suggestions: List[str]
    interview_questions: List[str]

    final_report: str
    next_step: Optional[str]


print("PlacementState created successfully!")

# cell 3 
def resume_analyzer_node(state: PlacementState):
    resume = state["resume"]

    prompt = f"""
You are a professional resume analyzer.

Analyze the following resume and extract the candidate's
technical skills, programming languages, frameworks, databases,
cloud tools, AI/ML tools, and other relevant job skills.

Return ONLY the skills, one skill per line.
Do not add numbering, explanations, or categories.

RESUME:
{resume}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    skills = [
        line.strip("-• ").strip()
        for line in content.split("\n")
        if line.strip()
    ]

    return {
        "candidate_skills": skills,
        "next_step": "job_analyzer"
    }


print("Resume Analyzer node created!")

#cell 4 
def job_analyzer_node(state: PlacementState):
    job_description = state["job_description"]

    prompt = f"""
You are a professional job-description analyzer.

Analyze the following job description and extract the skills
and technical requirements a candidate should have.

Include programming languages, frameworks, databases, cloud
technologies, AI/ML tools, software tools, and important
technical concepts.

Return ONLY the skills, one skill per line.
Do not add numbering, explanations, or categories.

JOB DESCRIPTION:
{job_description}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    skills = [
        line.strip("-• ").strip()
        for line in content.split("\n")
        if line.strip()
    ]

    return {
        "required_skills": skills,
        "next_step": "skill_matcher"
    }


print("Job Description Analyzer created!")

#cell 5 
def skill_matcher_node(state: PlacementState):
    candidate_skills = state["candidate_skills"]
    required_skills = state["required_skills"]

    candidate_lower = {skill.lower(): skill for skill in candidate_skills}

    matched_skills = []

    for skill in required_skills:
        if skill.lower() in candidate_lower:
            matched_skills.append(skill)

    return {
        "matched_skills": matched_skills,
        "next_step": "gap_analyzer"
    }

#cell6
def gap_analyzer_node(state: PlacementState):
    required_skills = state["required_skills"]
    matched_skills = state["matched_skills"]

    matched_lower = {skill.lower() for skill in matched_skills}

    missing_skills = [
        skill
        for skill in required_skills
        if skill.lower() not in matched_lower
    ]

    return {
        "missing_skills": missing_skills,
        "next_step": "improvement"
    }

#cell7
def improvement_node(state: PlacementState):
    missing_skills = state["missing_skills"]

    if not missing_skills:
        suggestions = [
            "The candidate currently matches all identified job requirements.",
            "Continue building practical AI/ML projects to strengthen hands-on experience.",
            "Prepare project explanations and technical interview questions."
        ]
    else:
        suggestions = [
            f"Learn and practice: {skill}"
            for skill in missing_skills
        ]

    return {
        "improvement_suggestions": suggestions,
        "next_step": "interview_generator"
    }

#cell 8
def interview_generator_node(state: PlacementState):
    missing_skills = state["missing_skills"]
    matched_skills = state["matched_skills"]

    questions = []

    # Questions based on matched skills
    for skill in matched_skills[:4]:
        questions.append(
            f"Can you explain your experience with {skill} and describe a project where you used it?"
        )

    # Questions based on skill gaps
    for skill in missing_skills[:4]:
        questions.append(
            f"What do you know about {skill}, and how would you learn or apply it in an AI/ML project?"
        )

    # General placement questions
    questions.extend([
        "Tell me about your most relevant AI or machine learning project.",
        "How would you approach building an AI application from idea to deployment?"
    ])

    return {
        "interview_questions": questions[:8],
        "next_step": "final_report"
    }
#cell9
def final_report_node(state: PlacementState):
    prompt = f"""
You are a professional placement advisor.

Create a clear placement-readiness report using the information below.

CANDIDATE SKILLS:
{state["candidate_skills"]}

REQUIRED JOB SKILLS:
{state["required_skills"]}

MATCHED SKILLS:
{state["matched_skills"]}

MISSING SKILLS:
{state["missing_skills"]}

IMPROVEMENT SUGGESTIONS:
{state["improvement_suggestions"]}

INTERVIEW QUESTIONS:
{state["interview_questions"]}

Format the report with these sections:

1. Candidate Skill Summary
2. Job Skill Requirements
3. Matching Skills
4. Skill Gaps
5. Improvement Plan
6. Interview Preparation

Keep the report factual and practical.
Do not invent information that is not present in the provided data.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return {
        "final_report": content,
        "next_step": "complete"
    }

print("Final Report node created!")

# cell 10
from langgraph.graph import StateGraph, START, END

placement_workflow = StateGraph(PlacementState)

placement_workflow.add_node("resume_analyzer", resume_analyzer_node)
placement_workflow.add_node("job_analyzer", job_analyzer_node)
placement_workflow.add_node("skill_matcher", skill_matcher_node)
placement_workflow.add_node("gap_analyzer", gap_analyzer_node)
placement_workflow.add_node("improvement", improvement_node)
placement_workflow.add_node("interview_generator", interview_generator_node)
placement_workflow.add_node("final_report", final_report_node)

placement_workflow.add_edge(START, "resume_analyzer")
placement_workflow.add_edge("resume_analyzer", "job_analyzer")
placement_workflow.add_edge("job_analyzer", "skill_matcher")
placement_workflow.add_edge("skill_matcher", "gap_analyzer")
placement_workflow.add_edge("gap_analyzer", "improvement")
placement_workflow.add_edge("improvement", "interview_generator")
placement_workflow.add_edge("interview_generator", "final_report")
placement_workflow.add_edge("final_report", END)

placement_app = placement_workflow.compile()

print("Placement Assistant LangGraph compiled successfully!")


#cell 11

sample_resume = """
Rahul is a Computer Science student with experience in Python,
Java, SQL, Git, and basic machine learning.

Projects:
- Built a student management system using Python and SQL.
- Created a machine learning project for predicting house prices.
- Used Git and GitHub for version control.

Skills:
Python, Java, SQL, Git, Machine Learning, Pandas, NumPy
"""

sample_job_description = """
We are looking for a Junior AI/ML Developer.

Required skills:
Python, Machine Learning, SQL, Pandas, NumPy,
LangChain, LangGraph, REST APIs, Git, and basic cloud knowledge.

The candidate should understand AI application development
and be able to build practical projects using modern AI tools.
"""

initial_state = {
    "resume": sample_resume,
    "job_description": sample_job_description,
    "required_skills": [],
    "candidate_skills": [],
    "matched_skills": [],
    "missing_skills": [],
    "improvement_suggestions": [],
    "interview_questions": [],
    "final_report": "",
    "next_step": "resume_analyzer"
}

#result = placement_app.invoke(initial_state)

#print("===== PLACEMENT ANALYSIS REPORT =====")
#print(result["final_report"])


#update
#print("Testing Gemini connection...")

#test_response = llm.invoke("Reply with exactly: Gemini test successful")

#print(test_response.content)