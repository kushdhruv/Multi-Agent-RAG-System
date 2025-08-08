from pydantic import BaseModel, Field
from typing import List

# NEW: A specific model for the ingestion request
class IngestRequest(BaseModel):
    documents: str = Field(
        ...,
        description="A URL pointing to the PDF document to be processed.",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?..."
    )

class RunRequest(BaseModel):
    """
    Defines the structure of the incoming request body for the /hackrx/run endpoint.
    """
    documents: str = Field(
        ..., 
        description="A URL pointing to the PDF document to be processed.",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?..."
    )
    questions: List[str] = Field(
        ..., 
        description="A list of questions to be answered based on the document.",
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases (PED)?"
        ]
    )

class RunResponse(BaseModel):
    """
    Defines the structure of the response body for the /hackrx/run endpoint.
    """
    answers: List[str] = Field(
        ...,
        description="A list of answers corresponding to the input questions.",
        example=[
            "A grace period of thirty days is provided for premium payment...",
            "There is a waiting period of thirty-six (36) months..."
        ]
    )
