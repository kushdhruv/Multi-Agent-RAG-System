from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from app.schemas.models import RunRequest
from app.core.security import verify_token
import asyncio
from typing import List
from cachetools import TTLCache
import hashlib
import os
import json

# Updated imports to use our tuned services
from app.services.three_retrieval_service import RetrievalService
from app.services.one_planning_synthesis_agent import PlanningSynthesisAgent

router = APIRouter()

# --- Centralized Configuration for Easy Tuning ---
# These parameters can be adjusted to trade off between accuracy and latency.
# Lower values generally lead to lower latency.
class AppConfig:
    POOL_TOP_K = 100  # Reduced candidate pool size
    GROUP_SIZE = 5  # Fewer centroids for hypothetical generation
    POOL_TOP_RERANK = 7  # Fewer chunks to rerank per question
    RETRY_EXPANDED_RERANK = 12  # Reduced number of extra chunks for retries
    MULTI_SUBQ_TOP_K = 25  # Reduced top_k for multi-sub-question retrieval
    SINGLE_SUBQ_TOP_K = 20  # Reduced top_k for single-question retrieval
    USE_SERVICE_SIDE_FALLBACK_IF_AVAILABLE = True
    CACHE_MAX_SIZE = 100 # Maximum number of items in the cache
    CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)

config = AppConfig()

retrieval_service: RetrievalService = None
combined_agent: PlanningSynthesisAgent = None
# Using TTLCache for a more robust caching strategy with size and time limits
question_cache = TTLCache(maxsize=config.CACHE_MAX_SIZE, ttl=config.CACHE_TTL)

def initialize_services():
    global retrieval_service, combined_agent
    if retrieval_service is None:
        print("Initializing RetrievalService & PlanningSynthesisAgent...")
        retrieval_service = RetrievalService(
            embedding_model_name="all-MiniLM-L6-v2",  # Faster model with smaller embedding dimension (384)
            reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            chunk_size_words=250,
            chunk_overlap_words=50,
            enable_lazy_cross_encoder=False
        )
        combined_agent = PlanningSynthesisAgent(retrieval_service=retrieval_service)
        print("Initialization complete.")

@router.post("/hackrx/run")
async def run_submission(request: RunRequest, _token: str = Depends(verify_token)):
        # --- ADDED: Log the incoming request body ---
    print("\n--- INCOMING REQUEST ---")
    try:
        # Use .model_dump_json for Pydantic v2+
        print(request.model_dump_json(indent=2))
    except AttributeError:
        # Fallback for Pydantic v1
        print(request.json(indent=2))
    print("------------------------\n")

    initialize_services()
    doc_url = request.documents
    m = hashlib.md5()
    m.update(doc_url.encode('utf-8'))
    doc_namespace = m.hexdigest()

    try:
        await asyncio.to_thread(
            retrieval_service.ingest_and_process_pdf,
            pdf_url=doc_url, 
            namespace=doc_namespace,
            force_reingest=False
        )
        question_cache.clear()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {e}"
        )

    try:
        # Step 1: Plans for each Q
        plans = [combined_agent.plan_and_research(q) for q in request.questions]
        all_hypos = [ha for plan in plans for ha_list in plan.values() for ha in ha_list]

        # Step 2: Global candidate pool
        candidate_pool = await asyncio.to_thread(
            retrieval_service.build_global_candidate_pool,
            all_hypos,
            config.POOL_TOP_K,
            config.GROUP_SIZE
        )

        # Step 3: Local rerank from pool
        def rerank_for(q: str) -> List[str]:
            return retrieval_service.rerank_from_pool(q, candidate_pool, top_n_rerank=config.POOL_TOP_RERANK)

        contexts_per_question = await asyncio.gather(*[
            asyncio.to_thread(rerank_for, q) for q in request.questions
        ])

        # Step 4: Batch synthesize
        answers = await combined_agent.synthesize_batch_answers(request.questions, contexts_per_question)

        # Step 5: Retry “Not found”
# --- Step 5: Retry "Not Found" Answers with Expanded Context ---
        # This is our second, more intensive attempt for questions that failed the first time.

        # Identify the original indices of all questions that returned "Not found"
        questions_to_retry_indices = [
            i for i, ans in enumerate(answers)
            if isinstance(ans, str) and "not found" in ans.lower()
        ]

        # Only proceed if there are questions to retry
        if questions_to_retry_indices:
            print(f"Retrying {len(questions_to_retry_indices)} questions with expanded context...")

            # Create a list of the question strings that need a retry
            retry_questions = [request.questions[i] for i in questions_to_retry_indices]
            
            # For these specific questions, retrieve a larger number of context chunks
            # from the original candidate pool. This is our new, more intensive strategy.
            expanded_contexts = [
                retrieval_service.rerank_from_pool(
                    q,
                    candidate_pool,
                    top_n_rerank=config.RETRY_EXPANDED_RERANK # Use a larger value for the retry
                ) for q in retry_questions
            ]

            # Send the new, smaller batch to the LLM for a second and final try
            retry_answers = await combined_agent.synthesize_batch_answers(
                retry_questions,
                expanded_contexts
            )

            # Update the original answers list with the new results from the retry attempt
            for original_index, new_answer in zip(questions_to_retry_indices, retry_answers):
                answers[original_index] = new_answer

         
        # --- ADDED: Log the final response ---
        print("\n--- FINAL ANSWERS RESPONSE ---")
        print(json.dumps({"answers": answers}, indent=2))
        print("----------------------------\n")
        # --- End of log ---

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        print(f"Batch error, falling back: {e}")
        answers = []
        for q in request.questions:
            try:
                answers.append(await run_single_question_pipeline(q))
            except Exception as inner:
                answers.append(f"[Error: {inner}]")
        return JSONResponse(content={"answers": answers})


async def run_single_question_pipeline(question: str) -> str:
    if question in question_cache:
        return question_cache[question]

    plan = combined_agent.plan_and_research(question)
    all_chunks = set()

    top_k = config.MULTI_SUBQ_TOP_K if len(plan) > 1 else config.SINGLE_SUBQ_TOP_K
    use_fallback = config.USE_SERVICE_SIDE_FALLBACK_IF_AVAILABLE and hasattr(retrieval_service, "search_and_rerank_with_fallback")

    async def get_ctx(sub_q: str, hypos: list):
        local = set()
        for ha in hypos:
            try:
                if use_fallback:
                    ctx = await asyncio.to_thread(retrieval_service.search_and_rerank_with_fallback, ha, top_k, config.POOL_TOP_RERANK)
                else:
                    ctx = await asyncio.to_thread(retrieval_service.search_and_rerank, ha, top_k)
                local.update(ctx)
            except Exception as e:
                print(f"Retrieval error: {e}")
        return local

    results = await asyncio.gather(*[get_ctx(sq, hypos) for sq, hypos in plan.items()])
    for r in results:
        all_chunks.update(r)

    answer = await combined_agent.synthesize_final_answer(question, list(all_chunks))
    question_cache[question] = answer
    return answer