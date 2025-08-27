import os
from typing import Optional

from langchain_community.vectorstores import Chroma

class DB:
  def __init__(self):
    self.CHROMA_PATH = "chroma"
    self.db: Optional[Chroma] = None

  def create_db(self, chunks, embedding):
    self.db = Chroma.from_documents(
      documents=chunks,
      embedding=embedding,
      persist_directory=self.CHROMA_PATH
    )

  def load_db(self, embedding):
    if os.path.exists(self.CHROMA_PATH):
      self.db = Chroma(
        embedding_function=embedding,
        persist_directory=self.CHROMA_PATH
      )
    else:
      raise FileNotFoundError(
                f"Persist dir '{self.CHROMA_PATH}' not found. "
                "Run ingestion to create the vector DB first."
            )

  def get_retriever(self, search_type: str = "similarity", k: int = 5):
    if self.db is None:
      raise RuntimeError("DB not loaded. Call load_db() first.")
    kwargs = {"k": k}
    if search_type == "similarity":
      return self.db.as_retriever(search_type="similarity", search_kwargs=kwargs)

  def close(self):
    self.db.close()
