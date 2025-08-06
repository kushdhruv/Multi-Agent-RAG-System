import google.generativeai as genai
from app.core.config import settings

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class HydeAgent:
    def __init__(self, api_key: str = settings.GOOGLE_API_KEY):
        # Set up the model configuration
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=250,
        )
        # Initialize the Gemini model
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=generation_config
        )

    def generate_hypothetical_answer(self, sub_question: str) -> str:
        """
        Generates a hypothetical, ideal answer for a given sub-question.
        This helps find documents that are conceptually similar, not just keyword matches.

        Args:
            sub_question: A simple, self-contained question.

        Returns:
            A dense, information-rich paragraph representing a hypothetical answer.
        """
        prompt = f"""
You are an expert in the domain of insurance policies. Your task is to generate a detailed, one-paragraph hypothetical answer to the following question. Imagine you have access to a perfect knowledge base.

Rules:
1. Do NOT start with phrases like "Based on the document..." or "I do not have access to...".
2. Directly write the most plausible and comprehensive answer you can invent.
3. The goal is to create a text that has high semantic similarity to a real, factual answer.

Question: "{sub_question}"
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in HyDE Agent: {e}")
            # If HyDE fails, we fall back to using the sub-question itself as the query.
            return sub_question