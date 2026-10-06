from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from config import smart
from models import JobRequirements, MatchResult, ComparisonResult
from extract import extract_job
from matcher import shortlist_candidates, evaluate_candidate as score_candidate, CV_FOLDER
import os

SHORTLIST_THRESHOLD = 60

comparator = smart.with_structured_output(ComparisonResult)

class ScreeningState(TypedDict):
    job_text: str                 
    job: JobRequirements          
    pool: list[str]               # CV filenames from RAG retrieval
    index: int                    # which candidate we're currently on
    results: list[MatchResult]    
    ranked: list[MatchResult]    
    shortlist: list[MatchResult]  
    best: MatchResult            
    best_reason: str              
    report: str                   


# NODES

def extract_job_requirements(state: ScreeningState) -> dict:

    return {"job": extract_job(state["job_text"])}

def retrieve_candidate_pool(state: ScreeningState) -> dict:

    pool = shortlist_candidates(state["job"])
    print(f"[retrieve] RAG shortlisted {len(pool)} CVs: {pool}")
    return {"pool": pool, "index": 0, "results": []}

def evaluate_candidate(state: ScreeningState) -> dict:

    i = state["index"]
    source = state["pool"][i]
    with open(os.path.join(CV_FOLDER, source), encoding="utf-8") as f:
        cv_text = f.read()
    result = score_candidate(state["job"], cv_text, source)
    print(f"[evaluate] {i + 1}/{len(state['pool'])}  {result.score:>3} | {result.candidate}")
    return {
        "results": state["results"] + [result],  # append, keep the list ourselves
        "index": i + 1,                           # move to the next candidate
    }

def rank(state: ScreeningState) -> dict:

    ranked = sorted(state["results"], key=lambda r: r.score, reverse=True)
    shortlist = [r for r in ranked if r.score >= SHORTLIST_THRESHOLD]
    provisional = ranked[0] if ranked else None
    print(f"[rank] {len(shortlist)} of {len(ranked)} cleared {SHORTLIST_THRESHOLD}")
    return {
        "ranked": ranked,
        "shortlist": shortlist,
        "best": provisional,
        "best_reason": provisional.reasoning if provisional else "No candidates found.",
    }

def compare(state: ScreeningState) -> dict:

    top = state["shortlist"][:3]
    summary = "\n\n".join(
        f"CANDIDATE: {r.candidate}\n"
        f"SCORE: {r.score}\n"
        f"STRENGTHS: {', '.join(r.matched)}\n"
        f"GAPS: {', '.join(r.missing)}\n"
        f"NOTE: {r.reasoning}"
        for r in top
    )
    prompt = (
        "You are a senior hiring manager making a final decision. "
        "Below are the top shortlisted candidates for this role, already scored. "
        "Weigh their strengths and gaps against each other and pick the single best hire. "
        f"Choose the winner by their exact name.\n\n"
        f"ROLE: {state['job'].title}\n\n{summary}"
    )
    decision = comparator.invoke(prompt)
    winner = next((r for r in top if r.candidate == decision.winner), top[0])
    print(f"[compare] head-to-head winner: {decision.winner}")
    return {"best": winner, "best_reason": decision.reasoning}

def report(state: ScreeningState) -> dict:

    lines = [f"\n=== SHORTLIST for: {state['job'].title} ===\n"]
    for rank_no, r in enumerate(state["ranked"], start=1):
        flag = "*" if r in state["shortlist"] else " "
        lines.append(f"{flag} {rank_no}. {r.score:>3} | {r.candidate}")
    if state["best"]:
        lines.append(f"\nRECOMMENDED HIRE: {state['best'].candidate}")
        lines.append(f"WHY: {state['best_reason']}")
    text = "\n".join(lines)
    return {"report": text}


# ROUTERS: these decide where to go next. They return a KEY, not state.

def decide(state: ScreeningState) -> str:

    if state["index"] < len(state["pool"]):
        return "evaluate"
    return "rank"

def should_compare(state: ScreeningState) -> str:

    if len(state["shortlist"]) >= 2:
        return "compare"
    return "report"


# WIRE THE GRAPH

def build_graph():
    g = StateGraph(ScreeningState)

    g.add_node("extract", extract_job_requirements)
    g.add_node("retrieve", retrieve_candidate_pool)
    g.add_node("evaluate", evaluate_candidate)
    g.add_node("rank", rank)
    g.add_node("compare", compare)
    g.add_node("report", report)

    g.add_edge(START, "extract")
    g.add_edge("extract", "retrieve")

    # The same router guards BOTH the loop entry and every iteration.
    g.add_conditional_edges("retrieve", decide, {"evaluate": "evaluate", "rank": "rank"})
    g.add_conditional_edges("evaluate", decide, {"evaluate": "evaluate", "rank": "rank"})

    g.add_conditional_edges("rank", should_compare, {"compare": "compare", "report": "report"})
    g.add_edge("compare", "report")
    g.add_edge("report", END)

    return g.compile()

def screen(job_text: str) -> ScreeningState:
    graph = build_graph()
    return graph.invoke({"job_text": job_text})

if __name__ == "__main__":
    job_text = """
    Backend Engineer. We need someone strong in Python and REST API design,
    with PostgreSQL and Docker. AWS and CI/CD experience are a plus.
    Minimum 2 years of experience.
    """
    final_state = screen(job_text)
    print(final_state["report"])
