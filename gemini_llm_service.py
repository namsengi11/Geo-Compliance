# gemini_llm_service.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmBlockThreshold, HarmCategory

class GeminiLLMService:
    def __init__(
        self,
        model_json: str = "gemini-2.5-flash",
        model_text: str = "gemini-2.5-flash",
        max_output_tokens: int = 1024,
        mode: str = "schema",
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

        # Plain text LLM for RAG (no forced JSON, no tools)
        self.llm = ChatGoogleGenerativeAI(model=model_json, **common)

        # Small helper you may already use elsewhere
        self._raw = ChatGoogleGenerativeAI(model=model_text, **{**common, "max_output_tokens": 512})

    def generate_text(self, prompt: str) -> str:
        msg = self._raw.invoke(prompt)
        return getattr(msg, "content", "") or ""
