"""
One Planning Synthesis Agent (Production Version v2)

Purpose:
- Produce a small number of high-quality, short hypotheticals to guide retrieval.
- Provide a targeted Gemini prompt builder that enforces a precise, expert-level response format.
- Connects to the live Gemini API to synthesize answers from context.
- Includes robust JSON parsing to handle variations in LLM output format.

Notes:
- This version requires the 'google-generativeai' library.
- Ensure your Gemini API key is set in your project's configuration.
"""
from typing import List, Optional, Any, Dict
import json
import asyncio
import re

# Import the Gemini library and your project's settings
import google.generativeai as genai
from app.core.config import settings


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    """Truncate a text to approximately max_words (preserves word boundaries)."""
    if not text:
        return text
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def generate_high_quality_hypotheticals(document_title: str, user_query: str, *, max_words_per_hypo: int = 35) -> List[str]:
    """
    Produce 3 hypotheticals (2 focused + 1 diverse) with strict brevity.
    These are used to expand recall when performing vector search.
    """
    q = user_query.replace("\n", " ").strip()
    title = document_title or "document"

    focused_1 = f'Direct, concise factual answer to: "{q}". If present in the document, include what the document explicitly states.'
    focused_2 = f'Step-by-step evidence or procedure in the document addressing: "{q}"; produce numbered items with short citations.'
    diverse = f'Limitations, caveats, or alternate views noted in the document about: "{q}". If none, infer likely limitations and label them "inferred".'

    h1 = _truncate_to_word_limit(focused_1, max_words_per_hypo)
    h2 = _truncate_to_word_limit(focused_2, max_words_per_hypo)
    h3 = _truncate_to_word_limit(diverse, max_words_per_hypo)

    return [h1, h2, h3]


def build_gemini_prompt(hypothetical: str, top_context_chunks: List[str], *, max_chunks: int = 4, chunk_char_limit: int = 800) -> str:
    """
    Creates a highly-directive, precise prompt for Gemini to act as an expert analyst.

    The prompt forces:
      - Use ONLY the provided context (no external knowledge).
      - A strict JSON object output with rich fields:
        {answer, answer_type, support, evidence, confidence, confidence_score, assumptions, follow_up_questions}
    """

    instruction = (
        "You are a world class domain expert and an evidence-focused document analyst. "
        "Your job is to answer the user's question using ONLY the provided context excerpts. "
        "Do NOT use outside knowledge, do NOT hallucinate, and do NOT guess beyond what the context shows. "
        "If the answer cannot be found or reasonably inferred from the context, return the exact string 'Not found in document' as the value for 'answer'."
    )

    # Build safe, truncated context blocks with headers. Include small tokens to make citations easier.
    safe_chunks = []
    for idx, c in enumerate(top_context_chunks[:max_chunks], start=1):
        if not c:
            continue
        chunk_trim = c if len(c) <= chunk_char_limit else c[:chunk_char_limit].rsplit(" ", 1)[0] + "..."
        # Retain an approximate page tag if present in the chunk (e.g. [PAGE 3])
        safe_chunks.append(f"[CONTEXT {idx}] {chunk_trim}")

    context = "\n\n---CONTEXT---\n\n" + "\n\n".join(safe_chunks) if safe_chunks else "\n\n---CONTEXT---\n\n[No context provided]"

    # Very explicit output schema to reduce parser errors & hallucinations
    format_instructions = (
        "\n\n---OUTPUT SCHEMA (MUST FOLLOW EXACTLY)---\n"
        "Return ONLY a single JSON object (no surrounding text, no markdown). The JSON MUST contain the following keys:\n\n"
        "1) 'answer' (string): A concise direct answer (1-2 sentences). If not found return exactly 'Not found in document'.\n\n"
        "2) 'answer_type' (string): one of ['fact', 'procedure', 'definition', 'opinion', 'mixed', 'not_found'] describing the nature of the answer.\n\n"
        "3) 'support' (array): an array of objects, each with { 'excerpt': <verbatim excerpt from context>, 'context_id': <CONTEXT N>, 'page_ref': <page if present or 'unknown'> }.\n"
        "   If no verbatim support exists, set support = ['none'].\n\n"
        "4) 'evidence' (array of strings): short numbered evidence bullets referencing which context id supports them. Example: ['(CONTEXT 2) Step 1 ...']\n\n"
        "5) 'confidence' (string): one of ['high','medium','low'] based strictly on how explicit the support is.\n\n"
        "6) 'confidence_score' (number): a decimal between 0.0 and 1.0 reflecting calibrated confidence.\n\n"
        "7) 'assumptions' (array of strings): list any assumptions you had to make. If none, return [].\n\n"
        "8) 'follow_up_questions' (array of strings): 0-3 short follow-up questions useful to clarify or expand the answer.\n\n"
        "IMPORTANT: Provide verbatim excerpts exactly as they appear in the provided context for 'support'. Page refs should be extracted from text if present (e.g. '[PAGE 3]'), otherwise 'unknown'.\n"
        "Do not include any additional keys or text. Return only the JSON object.\n"
    )

    prompt = f"{instruction}\n\n---TASK---\nBased on this hypothetical query: {hypothetical}\n{context}\n{format_instructions}"
    return prompt


def build_compact_batched_prompt(
    questions: List[str],
    contexts_per_question: List[List[str]],
    *,
    max_chunks_per_q: int = 5,
    per_chunk_char_limit: int = 500
) -> str:
    """
    Build a compact batched prompt that demands a JSON array of precise structured answers.
    Output will be a JSON array where each element is the object defined in build_gemini_prompt.
    """
    sections = []
    for i, (q, ctx) in enumerate(zip(questions, contexts_per_question), start=1):
        safe_ctx = []
        for c in (ctx or [])[:max_chunks_per_q]:
            if not c:
                continue
            ct = c if len(c) <= per_chunk_char_limit else c[:per_chunk_char_limit].rsplit(" ", 1)[0] + "..."
            safe_ctx.append(ct)
        ctx_block = "\n\n".join(safe_ctx) if safe_ctx else "[No context provided]"
        sections.append(f"--- QUESTION {i} ---\nQuestion: {q}\nContext:\n{ctx_block}")

    batched_prompt = (
        "You are a fast, highly accurate evidence-based QA system. For each question below, return "
        "a JSON object (as specified) with structured answer and evidence, using ONLY the context shown for that question.\n\n"
        "Return a single JSON array where each element corresponds to the respective question in order.\n\n"
        + "\n\n".join(sections)
        + "\n\nIMPORTANT: If an answer cannot be found for a question, its object 'answer' must be exactly 'Not found in document' and 'answer_type' must be 'not_found'."
    )
    return batched_prompt


class PlanningSynthesisAgent:
    def __init__(self, retrieval_service=None, model_name: str = 'gemini-2.5-flash'):
        self.retrieval_service = retrieval_service

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in settings. Please check your configuration.")

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        print(f"Gemini model '{model_name}' initialized successfully.")


    def generate_high_quality_hypotheticals(self, document_title, user_query, *, max_words_per_hypo=35):
        return generate_high_quality_hypotheticals(document_title, user_query, max_words_per_hypo=max_words_per_hypo)

    def build_gemini_prompt(self, hypothetical, top_context_chunks, *, max_chunks=4, chunk_char_limit=800):
        return build_gemini_prompt(hypothetical, top_context_chunks, max_chunks=max_chunks, chunk_char_limit=chunk_char_limit)

    def build_compact_batched_prompt(self, questions, contexts_per_question, *, max_chunks_per_q=5, per_chunk_char_limit=500):
        return build_compact_batched_prompt(questions, contexts_per_question, max_chunks_per_q=max_chunks_per_q, per_chunk_char_limit=per_chunk_char_limit)

    def plan_and_research(self, question, top_k: int = 5):
        """
        Minimal planning step: returns hypotheticals for the question.
        Upstream code drives retrieval; we keep this lightweight and deterministic.
        """
        if not self.retrieval_service:
            raise ValueError("RetrievalService not attached to PlanningSynthesisAgent")

        hypos = self.generate_high_quality_hypotheticals("document", question)
        # Return mapping: sub-question (here the original question) -> list of hypotheticals
        return {question: hypos}

    async def synthesize_batch_answers(self, questions: List[str], contexts_per_question: List[List[str]]) -> List[str]:
        """
        Builds a batched prompt, sends it to the Gemini API, and parses the response.
        Returns a list of short answer strings (extracted from the structured JSON 'answer' field).
        """
        if not questions:
            return []

        batched_prompt = self.build_compact_batched_prompt(questions, contexts_per_question)

        print("--- Calling Gemini API (batched) ---")
        try:
            response = await self.model.generate_content_async(batched_prompt)
            raw = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return [f"[Error: API call failed]" for _ in questions]

        # Robust JSON extraction: find the first JSON array in the output
        try:
            start = raw.find('[')
            end = raw.rfind(']')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON array found in LLM response.")
            json_str = raw[start:end+1]
            parsed = json.loads(json_str)

            # If the model returned a list of plain strings (legacy), keep them.
            if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                return parsed[:len(questions)]

            # If the model returned list of objects, extract 'answer' field for each.
            if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                answers = []
                for obj in parsed:
                    if not isinstance(obj, dict):
                        answers.append("[Error: Invalid item]")
                        continue
                    ans = obj.get("answer")
                    if isinstance(ans, str):
                        answers.append(ans)
                    else:
                        # fallback: try to stringify minimal info
                        answers.append(json.dumps(obj))
                # Ensure list length matches questions
                if len(answers) != len(questions):
                    print("Warning: parsed batch length differs from questions length.")
                    # pad/truncate to match
                    answers = (answers + ["[Error: Missing answer]"] * len(questions))[:len(questions)]
                return answers

            # Otherwise, return fallback error strings
            print("Warning: Unexpected JSON structure in batched response.")
            return [f"[Error: Unexpected batch response]" for _ in questions]

        except Exception as e:
            print(f"Warning: Failed to parse batched response JSON: {e}. Raw response:\n{raw}")
            return [f"[Error: Malformed LLM batch response]" for _ in questions]


    async def synthesize_final_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Builds a single prompt, sends it to the Gemini API, and returns the 'answer' string.
        """
        prompt = self.build_gemini_prompt(question, context_chunks)

        print("--- Calling Gemini API (single) ---")
        try:
            response = await self.model.generate_content_async(prompt)
            raw = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "[Error: API call failed]"

        # Clean up code fences and find JSON object
        try:
            # Find first JSON object in text (look for first '{' ... matching '}' at the end)
            start = raw.find('{')
            end = raw.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in LLM response.")
            json_str = raw[start:end+1]

            # Remove common markdown fences around the JSON if present
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)

            parsed = json.loads(json_str)
            # Prefer the main 'answer' field
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed.get("answer")
            # If the model returned legacy string, return it
            if isinstance(parsed, str):
                return parsed
            # Fallback to a compact representation
            return json.dumps(parsed)
        except Exception as e:
            print(f"Warning: Failed to decode structured JSON from LLM response: {e}. Raw response:\n{raw}")
            # As a safe fallback, try to extract any "Answer:" lines or return a portion of raw text
            m = re.search(r"Answer[:\-]\s*(.+)", raw, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # give a useful error string
            return "[Error: Failed to parse LLM response]"

