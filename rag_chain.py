from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema import Document

from compliance_prompt import compliance_prompt
from retriever_service import RetrieverService
from llm_service import LLMService

MAX_CONTEXT_CHARS = 6000 

@dataclass
class ComplianceResult:
    answer_text: str
    json: Dict[str, Any]
    documents: List[Document]

def build_rag_chain(
    retriever_service: RetrieverService,
    llm_service: LLMService
) -> RetrievalQA:
    prompt = compliance_prompt()
    llm = llm_service.llm

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_service,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        verbose=True # to see what is happening
    )
    return chain

def extract_json(raw: str) -> dict:
    # Drop code fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[-1] if parts else raw
    # Take the last {...} block
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("No JSON object found in input.")
    return json.loads(raw[s:e+1])


# def parser(result) -> ComplianceResult:
#     """
#     Parses the output from the RAG chain into a structured ComplianceResult.
#     Expects the output to have an 'Answer:' part and a JSON part.
#     """
#     full_text = result['result']
