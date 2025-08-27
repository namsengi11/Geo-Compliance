from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from db import DB

search_type = Literal["similarity"]

@dataclass
class Retrieved:
    doc: Document
    score: Optional[float] = None

class RetrieverService:
  def __init__(self,
        persist_dir: str = "chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        search_type: search_type = "similarity",
        k: int = 5):
    self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
    self.db = DB()
    self.db.load_db(self.embedding)

    self.retriever = self.db.get_retriever(
        search_type=search_type,
        k=k
    )  

  def retrieve(self, query: str) -> List[Document]:
    return self.retriever.get_relevant_documents(query)
  
  def retrieve_with_scores(self, query: str, k: int = 5) -> List[Retrieved]:
    vs = self.db.db
    if vs is None:
        raise RuntimeError("Vector store not loaded.")
    # This uses similarity search 
    pairs = vs.similarity_search_with_score(query, k=k)
    return [Retrieved(doc=d, score=s) for (d, s) in pairs]


  def close(self):
    self.db.close()