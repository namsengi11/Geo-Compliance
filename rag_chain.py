from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from compliance_prompt import compliance_prompt
from retriever_service import RetrieverService
from llm_service import LLMService
from gemini_llm_service import GeminiLLMService

# Fixed schema for Gemini structured output (NO citations)
class ComplianceSchema(BaseModel):
    compliance_needed: bool = Field(...)
    issues: List[str] = Field(..., max_items=5)   # ensure you keep issues concise in prompt
    reasoning: str = Field(...)
    jurisdiction: List[str] = Field(...)

@dataclass
class ComplianceResult:
    answer: Dict[str, Any] | str
    documents: List[Document]

def build_rag_chain(
    retriever_service: RetrieverService,
    llm_service: LLMService | GeminiLLMService,
):
    # Gemini → schema mode; Llama → text JSON mode
    is_gemini = isinstance(llm_service, GeminiLLMService)
    prompt = compliance_prompt(json_mode="schema" if is_gemini else "text")

    llm = llm_service.llm
    if is_gemini:
        # Enforce structured output (parsed dict returned in 'answer')
        try:
            # llm = llm.with_structured_output(ComplianceSchema)
            pass
        except Exception:
            # Fallback: still OK; Gemini will return JSON text due to response_mime_type
            pass

    # Stuff-docs chain (model-agnostic) + retrieval wrapper
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_service, doc_chain)
    return rag_chain  # returns dict with keys: input, context, answer

def run_rag_plaintext(llm, retriever, user_input: str) -> Dict[str, Any]:
    # 1) Retrieve
    docs: List[Document] = retriever.invoke(user_input)
    # Keep context compact to avoid truncation & safety surprises
    trimmed = []
    for d in docs[:4]:
        # hard cap each chunk to ~2000 chars to stay under limits
        pc = (d.page_content or "")[:2000]
        trimmed.append(pc.strip())
    context_text = "\n\n---\n".join([t for t in trimmed if t])

    # 2) Prompt (matches your existing variables: {input}, {context})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a geo-regulation compliance assistant. Use ONLY the provided Context. "
                   "If Context is insufficient, say you don't know."),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])
    prompt_value = prompt.invoke({"input": user_input, "context": context_text})

    # 3) Call Gemini and return the raw AIMessage so we can inspect metadata
    msg = llm.invoke(prompt_value)

    # 4) Always give a non-empty answer to the caller
    answer_text = (getattr(msg, "content", "") or "").strip()
    if not answer_text:
        # Fallback: at least tell us it was empty and attach metadata
        answer_text = "[Empty response from model]"
    return {
        "answer": answer_text,
        "context": docs,
        "response_metadata": getattr(msg, "response_metadata", {}),
    }