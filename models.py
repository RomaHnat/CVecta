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