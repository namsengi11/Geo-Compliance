import os

from langchain_community.vectorstores import Chroma

class DB:
  def __init__(self):
    self.CHROMA_PATH = "chroma"
    self.db = None

  def create_db(self, chunks, embedding):
    self.db = Chroma.from_documents(
      documents=chunks,
      embedding=embedding,
      persist_directory=self.CHROMA_PATH
    )

  def close(self):
    self.db.close()