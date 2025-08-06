from pydantic import BaseModel, Field
from typing import List

class RunRequest(BaseModel):
    """
    Defines the structure of the incoming request body for the /hackrx/run endpoint.
    Pydantic validates that the incoming JSON matches this schema.
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
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    )

class RunResponse(BaseModel):
    """
    Defines the structure of the response body for the /hackrx/run endpoint.
    FastAPI uses this model to serialize the output data into valid JSON.
    """
    answers: List[str] = Field(
        ...,
        description="A list of answers corresponding to the input questions.",
        example=[
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
        ]
    )