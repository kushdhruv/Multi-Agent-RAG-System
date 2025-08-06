import google.generativeai as genai
from typing import Dict
from app.core.config import settings

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class RefinerAgent:
    def __init__(self, api_key: str = settings.GOOGLE_API_KEY):
        # Set up the model configuration
        generation_config = genai.GenerationConfig(
            temperature=0.0,
        )
        # Initialize the Gemini model
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=generation_config
        )

    def refine_answer(self, draft_answer: str, critique_feedback: Dict) -> str:
        """
        Revises a draft answer based on the critic's feedback, if necessary.
        
        Args:
            draft_answer: The initial answer from the Synthesizer.
            critique_feedback: The JSON object from the Critic agent.

        Returns:
            The final, polished answer ready for the user.
        """
        is_supported = critique_feedback.get("is_supported", False)
        critique_text = critique_feedback.get("critique", "")

        # If the answer is supported and the critique is empty, no refinement is needed.
        if is_supported and not critique_text:
            return draft_answer

        prompt = f"""
You are a master editor. Your task is to revise the 'Original Draft Answer' based only on the provided 'Critique'.

Rules:
1. If the 'Critique' is empty, return the 'Original Draft Answer' exactly as it is.
2. If there is a critique, apply the suggested corrections to produce a final, improved answer.
3. Your output should only be the refined answer text, with no additional explanation.

---
Original Draft Answer:
{draft_answer}
---
Critique:
{critique_text}
---
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in Refiner Agent: {e}")
            # If refinement fails, return the original draft with a warning.
            return f" {draft_answer}"