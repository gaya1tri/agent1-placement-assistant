import streamlit as st
from app import placement_app
from pypdf import PdfReader
st.set_page_config(
    page_title="AI Placement Assistant",
    page_icon="🎯"
)

st.title("🎯 AI Placement Assistant")

st.write(
    "Upload a resume and provide a job description "
    "to analyze skills, gaps, and interview preparation."
)

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "txt"]
)

job_description = st.text_area(
    "Job Description",
    height=250
)

if st.button("Analyze Placement"):

    if resume_file is None:

        st.warning("Please upload a resume.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        pdf_reader = PdfReader(resume_file)
        resume_text = ""

        for page in pdf_reader.pages:
            resume_text += page.extract_text() or ""

        st.success("Resume text extracted successfully!")
        st.write("Resume text extracted successfully.")
        initial_state = {
            "resume": resume_text,
            "job_description": job_description,
            "required_skills": [],
            "candidate_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "improvement_suggestions": [],
            "interview_questions": [],
            "final_report": "",
            "next_step": "resume_analyzer"
        }
        st.write("Ready to analyze the placement profile.")
        result = placement_app.invoke(initial_state)
        st.write(result["final_report"])