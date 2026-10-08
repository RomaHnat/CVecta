from pydantic import BaseModel, Field

class CandidateProfile(BaseModel):
   
    full_name: str = Field(description="The candidate's full name")
    email: str | None = Field(default=None, description="Email address, if present")
    phone: str | None = Field(default=None, description="Phone number, if present")
    years_experience: float | None = Field(
        default=None,
        description="Approximate total years of professional work experience",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills, tools and technologies the candidate knows",
    )
    job_titles: list[str] = Field(
        default_factory=list,
        description="Job titles or roles the candidate has held",
    )
    summary: str | None = Field(
        default=None,
        description="A one or two sentence summary of the candidate",
    )

class JobRequirements(BaseModel):

    title: str = Field(description="The job title")
    required_skills: list[str] = Field(
        default_factory=list, description="Must-have skills and technologies"
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list, description="Preferred but optional skills"
    )
    min_years_experience: float | None = Field(
        default=None, description="Minimum years of experience required, if stated"
    )
    summary: str | None = Field(default=None, description="One-sentence role summary")


class MatchResult(BaseModel):

    candidate: str = Field(description="The candidate's name")
    source: str = Field(default="", description="The CV filename this result came from")
    score: int = Field(description="Match score from 0 to 100")
    matched: list[str] = Field(
        default_factory=list, description="Key requirements the candidate meets"
    )
    missing: list[str] = Field(
        default_factory=list, description="Key requirements the candidate lacks"
    )
    reasoning: str = Field(description="One or two sentence explanation of the score")

class ComparisonResult(BaseModel):
    
    winner: str = Field(description="The exact FILE value of the single best CV")
    reasoning: str = Field(description="Why this CV was chosen over the others")