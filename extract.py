from config import fast
from models import CandidateProfile

extractor = fast.with_structured_output(CandidateProfile)

def extract_candidate(cv_text: str) -> CandidateProfile:

    prompt = (
        "You are an expert recruiting assistant. Extract the candidate's "
        "information from the CV below. Only use information that is actually "
        "present — do not invent anything. If a field is missing, leave it empty.\n\n"
        f"CV:\n{cv_text}"
    )
    return extractor.invoke(prompt)

#quick manual test: runs only when you execute this file directly
if __name__ == "__main__":
    with open("cvs/samplecv.txt", "r", encoding="utf-8") as f:
        text = f.read()

    profile = extract_candidate(text)

    print("\n--- EXTRACTED PROFILE ---")
    print(f"Name:   {profile.full_name}")
    print(f"Email:  {profile.email}")
    print(f"Phone:  {profile.phone}")
    print(f"Years:  {profile.years_experience}")
    print(f"Titles: {', '.join(profile.job_titles)}")
    print(f"Skills: {', '.join(profile.skills)}")
    print(f"Summary:{profile.summary}")