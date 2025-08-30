# gemini_llm_service.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from transformers import ( pipeline
)

class GeminiLLMService:
    """
    llm: ChatGoogleGenerativeAI configured for schema (function-calling) mode.
         Do NOT set response_mime_type when using structured output.
    _raw: plain text helper for small utilities (classification, etc.).
    """
    def __init__(
        self,
        model_json: str = "gemini-2.5-flash",
        model_text: str = "gemini-2.5-flash",
        max_output_tokens: int = 8192,
    ):
        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        common = dict(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            safety_settings=safety,
            google_api_key=os.environ["GEMINI_API_KEY"],
            convert_system_message_to_human=True,
        )

        # Main LLM (no response_mime_type, no response_schema here)
        self.llm = ChatGoogleGenerativeAI(model=model_json, response_mime_type="application/json",  **{**common, "max_output_tokens": max_output_tokens})
        # Helper for quick plain-text prompts
        self._raw = ChatGoogleGenerativeAI(model=model_text, **{**common, "max_output_tokens": 4096})

    def generate_text(self, prompt: str) -> str:
        msg = self._raw.invoke(prompt)
        return getattr(msg, "content", "") or ""
