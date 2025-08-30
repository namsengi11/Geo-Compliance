# rag_chain.py
from __future__ import annotations
import json
from typing import Any, Dict, List
from pydantic import Field

from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from compliance_prompt import compliance_prompt
from terminology import expand_query


class ExpandedFilteredRetriever(BaseRetriever):
    """
    A compliant LangChain retriever that:
      - expands acronyms/codenames in the query before retrieval
      - filters out glossary docs (metadata.doc_type == "glossary") so they
        never get stuffed into the prompt
    """
    base: BaseRetriever | Any = Field(repr=False)

    def _strip_glossary(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        return [
            d for d in docs
            if (getattr(d, "metadata", None) or {}).get("doc_type") != "glossary"
        ]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        q = expand_query(query)
        # Support both BaseRetriever and simple services with the same method
        if isinstance(self.base, BaseRetriever):
            docs = self.base.get_relevant_documents(q)
        else:
            docs = self.base.get_relevant_documents(q)  # best-effort delegation
        return self._strip_glossary(docs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        q = expand_query(query)
        if isinstance(self.base, BaseRetriever):
            docs = await self.base.aget_relevant_documents(q)
        else:
            # Fallback to sync path if async not available
            docs = self.base.get_relevant_documents(q)
        return self._strip_glossary(docs)


def build_rag_chain(
    retriever,         # can be a BaseRetriever OR your custom service
    llm_service,       # GeminiLLMService or LLMService (must expose .llm)
) -> RetrievalQA:
    prompt = compliance_prompt()

    # Wrap the provided retriever with our BaseRetriever-compatible wrapper
    wrapped = ExpandedFilteredRetriever(base=retriever)

    qa = RetrievalQA.from_chain_type(
        llm=llm_service.llm,
        chain_type="stuff",
        retriever=wrapped,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
        verbose=False,
    )
    return qa


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
