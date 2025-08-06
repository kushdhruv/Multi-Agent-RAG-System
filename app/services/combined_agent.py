import json
import google.generativeai as genai
from typing import List, Dict
from app.core.config import settings

# Configure the Gemini client
genai.configure(api_key=settings.GOOGLE_API_KEY)

class CombinedAgent:
    """
    An efficient agent that combines the Synthesizer and Critic roles.
    It generates a draft answer and simultaneously critiques it for factual accuracy
    in a single LLM call, returning a structured JSON object.
    """
    def __init__(self):
        # Set up the model configuration for JSON output
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
        )
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash", # A powerful and fast model suitable for this task
            generation_config=generation_config
        )

    def synthesize_and_critique(self, original_question: str, context_chunks: List[str]) -> Dict:
        """
        Generates a draft answer and critiques it against the source text.
        
        Args:
            original_question: The user's original, complete question.
            context_chunks: A list of high-quality, relevant text chunks from the document.

        Returns:
            A dictionary containing the answer, its support status, and a critique.
        """
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
You are a precise and factual writing assistant with a built-in fact-checker. Your task is to answer the user's question based ONLY on the provided 'Source Text' and then critique your own answer for accuracy.

Your output MUST be a single JSON object with the following three keys: "answer", "is_supported", and "critique".

Rules for the "answer" value:
1. Synthesize the information from the 'Source Text' into a comprehensive answer.
2. If the information is not present, state that the answer could not be found in the document.
3. Do not use any external knowledge or make assumptions.

Rules for the "is_supported" and "critique" values:
1. After generating the answer, critically review it.
2. Set "is_supported" to true if every single statement in your generated answer is directly supported by the 'Source Text'. Otherwise, set it to false.
3. In the "critique" field, provide a brief explanation if there are any unsupported statements or missing information. If the answer is perfect, this should be an empty string.

---
Source Text:
{context_str}
---
User's Original Question:
"{original_question}"
---
"""

        try:
            response = self.model.generate_content(prompt)
            # The response text will be a JSON string
            response_data = json.loads(response.text)
            
            # Ensure the output has the expected keys, with safe defaults
            return {
                "answer": response_data.get("answer", "Error: Could not generate an answer."),
                "is_supported": response_data.get("is_supported", False),
                "critique": response_data.get("critique", "Agent failed to produce a valid critique.")
            }

        except Exception as e:
            print(f"Error in Combined Agent: {e}")
            # Return a structured error response
            return {
                "answer": "An error occurred during the synthesis and critique process.",
                "is_supported": False,
                "critique": f"An exception occurred: {str(e)}"
            }
