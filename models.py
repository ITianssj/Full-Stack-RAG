# models.py — Smart validation (never crashes the app)
from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question")
    collection: Optional[str] = "default"
    top_k: Optional[int] = Field(8, ge=1, le=20)

    # This runs automatically on creation
    def model_post_init(self, __context):
        # Clean and validate question
        q = self.question.strip()
        if len(q) == 0:
            raise ValueError("Question cannot be empty")
        if len(q) < 3:
            # Auto-fix short inputs instead of crashing
            self.question = q + " (please answer in detail)"
        else:
            self.question = q

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to document")
    collection: Optional[str] = "default"