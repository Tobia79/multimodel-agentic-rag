from typing import List, Literal
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    route: Literal["direct", "rag", "clarify"] = Field(
        description="Answer path: direct (no retrieval), rag (search documents), or clarify.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision.",
    )
    reason: str = Field(
        description="Brief explanation for the routing choice.",
    )
    clarification_needed: str = Field(
        default="",
        description="Question to ask the user when route is clarify.",
    )


class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )


class RetrievalConfidenceAssessment(BaseModel):
    confidence: int = Field(
        ge=0,
        le=10,
        description="Retrieval quality score from 0 (useless) to 10 (excellent).",
    )
    reasoning: str = Field(
        description="Brief explanation for the confidence score.",
    )