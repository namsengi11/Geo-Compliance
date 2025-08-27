import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from document_loader import DocumentLoader
from db import DB

class DocumentManager:
  def __init__(self, dir):
    self.dir = dir
    self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

  def split_docs_to_chunks(self, chunk_size=1000, chunk_overlap=500):
    docs = []
    for doc in os.listdir(self.dir):
      try:
        loader = DocumentLoader(os.path.join(self.dir, doc))
        docs.extend(loader.load())
      except Exception as e:
        print(f"Error loading {doc}: {e}")
        continue

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      add_start_index=True
    )

    chunks = text_splitter.split_documents(docs)
    return chunks

  def save_to_db(self, chunks):
    embedding = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
    db = DB()
    db.create_db(chunks, embedding)

if __name__ == "__main__":
  manager = DocumentManager("./regulations/")
  chunks = manager.split_docs_to_chunks()
  manager.save_to_db(chunks)

