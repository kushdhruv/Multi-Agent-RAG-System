from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.models import RunRequest, RunResponse
from app.core.security import verify_token
import asyncio

# Correctly import the CLASS from each service file
from app.services.three_retrieval_service import RetrievalService
from app.services.one_decomposer_agent import DecomposerAgent
from app.services.two_hyde_agent import HydeAgent
from app.services.combined_agent import CombinedAgent # Import the new merged agent
from app.services.six_refiner_agent import RefinerAgent

# Create an API router
router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=RunResponse,
    summary="Run the full document QA pipeline"
)
async def run_submission(
    request: RunRequest,
    # This dependency protects the endpoint by verifying the Bearer token
    _token: str = Depends(verify_token) 
):
    """
    This endpoint orchestrates the entire multi-agent RAG workflow.
    
    - **Authentication**: Requires a valid Bearer token.
    - **Ingestion**: Downloads and processes the PDF from the provided URL.
    - **Orchestration**: For each question, it runs a more efficient agent pipeline:
      1. Decompose -> 2. HyDE -> 3. Retrieve/Rerank -> 4. Synthesize & Critique -> 5. Refine
    - **Response**: Returns a list of final, fact-checked answers.
    """
    try:
        # Initialize the retrieval service. This loads the ML models.
        retrieval_service = RetrievalService()
        
        # Ingest and process the PDF. This is a blocking I/O and CPU-bound task.
        # Running in a threadpool to avoid blocking the main async event loop.
        await asyncio.to_thread(retrieval_service.ingest_and_process_pdf, request.documents)

        final_answers = []
        
        # Initialize agent instances
        decomposer = DecomposerAgent()
        hyde = HydeAgent()
        combined_agent = CombinedAgent()
        refiner = RefinerAgent()

        # Process each question sequentially through the agent pipeline
        for question in request.questions:
            print(f"\n--- Processing Question: {question} ---")

            # 1. Decomposer Agent
            sub_questions = decomposer.decompose_question(question)
            print(f"Decomposed into: {sub_questions}")

            # 2. HyDE Agent and 3. Retrieval Service
            all_context_chunks = set() # Use a set to avoid duplicate context
            for sub_q in sub_questions:
                hypothetical_answer = hyde.generate_hypothetical_answer(sub_q)
                print(f"  - Hypothetical answer for '{sub_q}': '{hypothetical_answer[:50]}...'")
                
                # Use the hypothetical answer to find the best context
                context = retrieval_service.search_and_rerank(hypothetical_answer)
                all_context_chunks.update(context)
            
            print(f"Retrieved {len(all_context_chunks)} unique context chunks.")

            # 4. Combined Synthesizer and Critic Agent
            combined_output = combined_agent.synthesize_and_critique(question, list(all_context_chunks))
            draft_answer = combined_output.get("answer", "Failed to generate an answer.")
            critique = {
                "is_supported": combined_output.get("is_supported", False),
                "critique": combined_output.get("critique", "Agent failed to produce a critique.")
            }
            print(f"Synthesized draft: '{draft_answer[:100]}...'")
            print(f"Simultaneous Critique: {critique}")

            # 5. Refiner Agent
            final_answer = refiner.refine_answer(draft_answer, critique)
            print(f"Final Answer: '{final_answer[:100]}...'")
            
            final_answers.append(final_answer)

        return RunResponse(answers=final_answers)

    except Exception as e:
        print(f"An unexpected error occurred in the main workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )
