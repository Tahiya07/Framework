from pydantic import BaseModel, Field
from typing import Literal

class LoginRequest(BaseModel): role: Literal["student", "teacher"]; access_code: str = Field(min_length=1, max_length=200)
class TextRequest(BaseModel): question: str = Field(min_length=1, max_length=4000); scope: Literal["public", "protected"] = "public"; top_k: int = Field(default=4, ge=1, le=8)
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)
class ChatRequest(TextRequest):
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    summary: bool = False
class IndexTextRequest(BaseModel): text: str = Field(min_length=1, max_length=200000); name: str = Field(default="pasted-text", max_length=150); scope: Literal["public", "protected"] = "public"; content_type: str = Field(default="study_material", max_length=40)
class BloomRequest(BaseModel): question: str = Field(min_length=1, max_length=4000)
class ModerationReviewRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    decision: Literal["approved", "needs_revision", "rejected"]
    notes: str = Field(default="", max_length=2000)
