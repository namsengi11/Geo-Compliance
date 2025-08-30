import chromadb
import os

from langchain_community.vectorstores import Chroma

class DB:
  def __init__(self, embedding):
    self.CHROMA_BASE_PATH = "chroma"
    self.db_path = self.CHROMA_BASE_PATH
    self.db: Chroma = Chroma(
      embedding_function=embedding,
      persist_directory=self.db_path
    )


  def insert_chunks(self, chunks):
    self.db.add_documents(chunks)

  def get_retriever(self, search_type: str = "similarity", region: str | None = None, k: int = 5):
    if self.db is None:
      raise RuntimeError("DB not loaded. Call load_db() first.")
    kwargs = {"k": k}
    if search_type == "similarity":
      if region and region.lower() != "global":
        kwargs["filter"] = {"region": region}
      return self.db.as_retriever(search_type="similarity", search_kwargs=kwargs)

  def close(self):
    self.db.close()
