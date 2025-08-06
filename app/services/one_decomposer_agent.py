import json
from typing import List
from app.core.config import settings
import google.generativeai as genai

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class DecomposerAgent:

    def __init__(self):
        # Set up the model configuration for JSON output if needed
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
        )
        # Initialize the Gemini model
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=generation_config
        )

    def decompose_question(self,question: str) -> List[str]:
        """
        Uses an LLM to decompose a complex question into simpler sub-questions.
        
        Args:
            question: The user's original, potentially complex question.

        Returns:
            A list of simple, self-contained sub-questions.
        """
        prompt = f"""
    You are a master of logical reasoning and query planning. Your task is to decompose a complex user question into a series of simpler, self-contained sub-questions that can be answered independently.

    Rules:
    1. If the question is already simple and self-contained, return it as a single-element array.
    2. Ensure each sub-question is a full, grammatical question.
    3. Your output MUST be a JSON array of strings. Do not add any other text or explanation.

    User Question: "{question}"
    """

        try:
            response = self.model.generate_content(prompt)
            response_data = json.loads(response.text)
            
            # --- FIX STARTS HERE ---
            # Check if the response is already a list
            if isinstance(response_data, list):
                return response_data
            
            # If it's a dictionary, find the list within it
            if isinstance(response_data, dict):
                for value in response_data.values():
                    if isinstance(value, list):
                        return value
            # --- FIX ENDS HERE ---
            
            # Fallback if the structure is unexpected
            print(f"Warning: Decomposer returned unexpected JSON structure: {response_data}")
            return [question]

        except Exception as e:
            print(f"Error in Gemini Decomposer Agent: {e}")
            return [question]