from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from config import smart
from models import JobRequirements, MatchResult, ComparisonResult
from extract import extract_job
from matcher import shortlist_candidates, evaluate_candidate as score_candidate, CV_FOLDER
import os
from loaders import read_cv_text, is_readable
from ocr import ocr_pdf, OCR_AVAILABLE

SHORTLIST_THRESHOLD = 60

comparator = smart.with_structured_output(ComparisonResult)

class ScreeningState(TypedDict):
    job_text: str                 
    job: JobRequirements          
    pool: list[str]               # CV filenames from RAG retrieval
    index: int                    # which candidate we're currently on
    current_source: str
    current_text: str
    results: list[MatchResult]
    unprocessed: list[str]    
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
    return {"pool": pool, "index": 0, "results": [], "unprocessed": []}

def load_candidate(state: ScreeningState) -> dict:

    i = state["index"]
    source = state["pool"][i]
    try:
        text = read_cv_text(os.path.join(CV_FOLDER, source))
    except Exception:
        text = ""
    print(f"[load] {i + 1}/{len(state['pool'])} {source} -> {len(text.strip())} chars")
    return {"current_source": source, "current_text": text}

def ocr_candidate(state: ScreeningState) -> dict:

    source = state["current_source"]
    print(f"[ocr] light read failed for {source}, trying Tesseract OCR...")
    return {"current_text": ocr_pdf(os.path.join(CV_FOLDER, source))}

def score_candidate_node(state: ScreeningState) -> dict:

    source = state["current_source"]
    result = score_candidate(state["job"], state["current_text"], source)
    print(f"[score] {result.score:>3} | {result.candidate} [{source}]")
    return {"results": state["results"] + [result], "index": state["index"] + 1}

def skip_candidate(state: ScreeningState) -> dict:

    source = state["current_source"]
    reason = "Tesseract OCR not installed" if not OCR_AVAILABLE else "unreadable even after OCR"
    print(f"[skip] {source} not processed ({reason})")
    return {"unprocessed": state["unprocessed"] + [source], "index": state["index"] + 1}

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
        f"FILE: {r.source}\n"
        f"CANDIDATE: {r.candidate}\n"
        f"SCORE: {r.score}\n"
        f"STRENGTHS: {', '.join(r.matched)}\n"
        f"GAPS: {', '.join(r.missing)}\n"
        f"NOTE: {r.reasoning}"
        for r in top
    )
    prompt = (
        "You are a senior hiring manager making a final decision. "
        "Below are the top shortlisted CVs for this role, already scored. "
        "Weigh their strengths and gaps and pick the single best one. "
        "Identify the winner by its exact FILE value.\n\n"
        f"ROLE: {state['job'].title}\n\n{summary}"
    )
    decision = comparator.invoke(prompt)
    winner = next((r for r in top if r.source == decision.winner), top[0])
    return {"best": winner, "best_reason": decision.reasoning}

def report(state: ScreeningState) -> dict:

    lines = [f"\n=== SHORTLIST for: {state['job'].title} ===\n"]
    for rank_no, r in enumerate(state["ranked"], start=1):
        flag = "*" if r in state["shortlist"] else " "
        lines.append(f"{flag} {rank_no}. {r.score:>3} | {r.candidate}  [{r.source}]")
    if state["best"]:
        lines.append(f"\nRECOMMENDED HIRE: {state['best'].candidate}  [{state['best'].source}]")
        lines.append(f"WHY: {state['best_reason']}")
    if state["unprocessed"]:
        reason = ("Tesseract OCR is not installed" if not OCR_AVAILABLE
                  else "they could not be read even after OCR")
        lines.append(f"\nNOT PROCESSED ({reason}):")
        lines += [f"  - {src}" for src in state["unprocessed"]]
    return {"report": "\n".join(lines)}


# ROUTERS: these decide where to go next. They return a KEY, not state.

def decide(state: ScreeningState) -> str:

    if state["index"] < len(state["pool"]):
        return "load"
    return "rank"

def route_after_load(state: ScreeningState) -> str:
    if is_readable(state["current_text"]):
        return "score"
    # light read failed — only PDFs can be rescued by OCR, and only if it's available
    if state["current_source"].lower().endswith(".pdf") and OCR_AVAILABLE:
        return "ocr"
    return "skip"

def route_after_ocr(state: ScreeningState) -> str:
    return "score" if is_readable(state["current_text"]) else "skip"

def should_compare(state: ScreeningState) -> str:

    if len(state["shortlist"]) >= 2:
        return "compare"
    return "report"


# WIRE THE GRAPH

def build_graph():
    g = StateGraph(ScreeningState)

    for name, fn in [
        ("extract", extract_job_requirements), ("retrieve", retrieve_candidate_pool),
        ("load", load_candidate), ("ocr", ocr_candidate),
        ("score", score_candidate_node), ("skip", skip_candidate),
        ("rank", rank), ("compare", compare), ("report", report),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "extract")
    g.add_edge("extract", "retrieve")

    # The same router guards BOTH the loop entry and every iteration.
    g.add_conditional_edges("retrieve", decide, {"load": "load", "rank": "rank"})
    g.add_conditional_edges("load", route_after_load, {"score": "score", "ocr": "ocr", "skip": "skip"})
    g.add_conditional_edges("ocr", route_after_ocr, {"score": "score", "skip": "skip"})
    g.add_conditional_edges("score", decide, {"load": "load", "rank": "rank"})
    g.add_conditional_edges("skip", decide, {"load": "load", "rank": "rank"})
    g.add_conditional_edges("rank", should_compare, {"compare": "compare", "report": "report"})
    g.add_edge("compare", "report")
    g.add_edge("report", END)

    return g.compile()

def screen(job_text: str) -> ScreeningState:
    return build_graph().invoke(job_text and {"job_text": job_text})

if __name__ == "__main__":
    job_text = """
    Backend Engineer. We need someone strong in Python and REST API design,
    with PostgreSQL and Docker. AWS and CI/CD experience are a plus.
    Minimum 2 years of experience.
    """
    final_state = screen(job_text)
    print(final_state["report"])
