from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from pydantic import Field
from langchain_core.runnables import RunnableConfig

from db import DB

search_type = Literal["similarity"]

@dataclass
class Retrieved:
    doc: Document
    score: Optional[float] = None

class RetrieverService(BaseRetriever):
  embedding: Optional[HuggingFaceEmbeddings] = Field(default=None, exclude=True)
  retriever: Dict[str, VectorStoreRetriever] = Field(default_factory=dict, exclude=True)
  
  def __init__(self,
        embedding: HuggingFaceEmbeddings,
        search_type: search_type = "similarity",
        db_name: str = "",
        retriever: Dict[str, VectorStoreRetriever] = dict(),
        k: int = 5,
        ):

    # Initialize parent class
    super().__init__()

    self.embedding = embedding
    
    if not retriever:
      try:
        db = DB(db_name, self.embedding)
        self.retriever[db_name] = db.get_retriever(
            search_type=search_type,
            k=k
        )
      except Exception as e:
        raise FileNotFoundError(f"Database {db_name} not found or failed to load. Please run ingestion to create the vector DB first. Error: {e}")
    else:
      self.retriever = retriever

  def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
    result = []
    for region, retriever in self.retriever.items():
      try:
        # Use get_relevant_documents instead of invoke for vector store retrievers
        docs = retriever.invoke(query)
        result.extend(docs)  # Flatten the list of documents
      except Exception as e:
        print(f"Error retrieving from {region}: {e}")
        raise
    print(result)
    return result

  # def retrieve(self, query: str) -> List[Document]:
  #   # Since self.retriever is a dict, we need to aggregate results from all retrievers
  #   result = []
  #   for retriever in self.retriever.values():
  #     docs = retriever.get_relevant_documents(query)
  #     result.extend(docs)
  #   return result
