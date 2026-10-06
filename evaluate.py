import sys
from dataclasses import dataclass, field

from graph import screen, SHORTLIST_THRESHOLD

@dataclass
class EvalCase:
    name: str                                      
    job_text: str                                   
    expected_winner: str | None = None              
    expected_top: list[str] = field(default_factory=list)  
    top_k: int = 3                                  
    should_not_shortlist: list[str] = field(default_factory=list)  

def _name_matches(actual: str, expected: str) -> bool:

    a, e = actual.strip().lower(), expected.strip().lower()
    return a == e or e in a or a in e

def run_case(case: EvalCase) -> bool:

    state = screen(case.job_text)
    ranked = state["ranked"]
    shortlist = state["shortlist"]
    best = state["best"]

    names_ranked = [r.candidate for r in ranked]
    names_shortlist = [r.candidate for r in shortlist]

    print(f"\n--- {case.name} ---")
    print("ranking:", " | ".join(f"{r.score} {r.candidate}" for r in ranked))
    if best:
        print("chosen best:", best.candidate)

    checks: list[tuple[str, bool]] = []

    # 1. The compare node picked the right person.
    if case.expected_winner is not None:
        ok = best is not None and _name_matches(best.candidate, case.expected_winner)
        checks.append((f"winner is {case.expected_winner}", ok))

    # 2. Each expected candidate shows up in the top K.
    for name in case.expected_top:
        top_names = names_ranked[: case.top_k]
        ok = any(_name_matches(n, name) for n in top_names)
        checks.append((f"{name} in top {case.top_k}", ok))

    # 3. Each mismatched candidate stayed OUT of the shortlist.
    for name in case.should_not_shortlist:
        ok = not any(_name_matches(n, name) for n in names_shortlist)
        checks.append((f"{name} NOT shortlisted (< {SHORTLIST_THRESHOLD})", ok))

    for label, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {label}")

    return all(ok for _, ok in checks)


# TEST CASES

CASES: list[EvalCase] = [
    EvalCase(
        name="Python backend role",
        job_text=(
            "Backend Engineer. Strong Python and REST API design, PostgreSQL, Docker. "
            "AWS and CI/CD a plus. Minimum 2 years."
        ),
        expected_top=["Alex Morgan"],
        should_not_shortlist=["Clair Connel"],
    ),
    EvalCase(
        name="Cloud / AWS role",
        job_text=(
            "Cloud Engineer. Deep AWS (EC2, S3, IAM, VPC, RDS), Terraform, Kubernetes, "
            "CI/CD pipelines, Linux. Minimum 3 years."
        ),
        expected_top=["James Taylor"],
    ),
    EvalCase(
        name="Java / JVM role",
        job_text=(
            "Software Engineer. Java and Spring Boot, microservices, Kotlin a plus, "
            "REST APIs, PostgreSQL. Minimum 3 years."
        ),
        expected_top=["Daniel Brooks"],
    ),
]


if __name__ == "__main__":
    results = [(c.name, run_case(c)) for c in CASES]

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n========== {passed}/{total} cases passed ==========")
    for name, ok in results:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")

    # Non-zero exit on any failure, so this behaves like a real test in CI.
    sys.exit(0 if passed == total else 1)