import os

from config import fast
from models import MatchResult
from extract import extract_job
from vectorstore import get_vector_store

CV_FOLDER = "cvs"

match_evaluator = fast.with_structured_output(MatchResult)


def build_query(job) -> str:

    parts = [job.title or ""] + job.required_skills + job.nice_to_have_skills
    return " ".join(parts)


def shortlist_candidates(job, k: int = 8, max_candidates: int = 5) -> list[str]:

    db = get_vector_store()
    chunks = db.similarity_search(build_query(job), k=k)
    seen = []
    for c in chunks:                       # most-similar first
        src = c.metadata["source"]
        if src not in seen:
            seen.append(src)
    return seen[:max_candidates]


def evaluate_candidate(job, cv_text: str, source: str) -> MatchResult:

    prompt = (
        "You are an expert technical recruiter. Evaluate how well the candidate "
        "matches the job. Be fair and evidence-based; score 0-100.\n\n"
        f"JOB TITLE: {job.title}\n"
        f"REQUIRED SKILLS: {', '.join(job.required_skills)}\n"
        f"NICE TO HAVE: {', '.join(job.nice_to_have_skills)}\n"
        f"MIN YEARS: {job.min_years_experience}\n\n"
        f"CANDIDATE CV ({source}):\n{cv_text}"
    )
    return match_evaluator.invoke(prompt)


def screen(job_text: str):

    job = extract_job(job_text)
    sources = shortlist_candidates(job)

    results = []
    for src in sources:
        with open(os.path.join(CV_FOLDER, src), encoding="utf-8") as f:
            cv_text = f.read()
        results.append(evaluate_candidate(job, cv_text, src))

    results.sort(key=lambda r: r.score, reverse=True)   # best first
    return job, results


if __name__ == "__main__":
    job_text = """
    Backend Software Engineer
    We need a backend engineer with strong Python and REST API experience.
    Must have: Python, REST APIs, PostgreSQL, Docker.
    Nice to have: AWS, CI/CD, Redis.
    Minimum 3 years of experience.
    """

    job, results = screen(job_text)

    print(f"\nJOB: {job.title}")
    print(f"Required: {', '.join(job.required_skills)}\n")
    print("--- RANKED CANDIDATES ---")
    for r in results:
        print(f"{r.score:3d} | {r.candidate}")
        print(f"      matched: {', '.join(r.matched[:4])}")
        print(f"      missing: {', '.join(r.missing[:4])}")
        print(f"      {r.reasoning}\n")