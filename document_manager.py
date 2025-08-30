# document_manager.py
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document  # NEW

from document_loader import DocumentLoader
from db import DB
from terminology import GLOSSARY  # NEW: use your glossary dict

class DocumentManager():
  def __init__(self, dir):
    self.dir = dir
    self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    self.embedding = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)

  def process_documents(self, chunk_size=1000, chunk_overlap=500):
    '''
    Chunks the documents and saves them to the database as embeddings.
    Takes regional metadata from texts-available.csv.
    Saves the embeddings to the database (single collection) with region metadata.
    '''

    with open("texts-available.csv", "r") as f:
      _ = f.readline()            # skip header
      texts_available = f.readlines()

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      add_start_index=True
    )

    for row in texts_available:
      if not row.strip():
        continue
      region, text_name = [s.strip() for s in row.split(",", 1)]
      try:
        loader = DocumentLoader(os.path.join(self.dir, text_name))
        document = loader.load()
      except Exception as e:
        print(f"Error loading {text_name}: {e}")
        continue

      chunks = text_splitter.split_documents(document)
      self.save_to_db(region, chunks)

    self.save_glossary_to_db()

  def save_to_db(self, region, chunks):
    for d in chunks:
      if not getattr(d, "metadata", None):
        d.metadata = {}
      d.metadata["region"] = region.strip()

    db = DB(self.embedding)
    db.insert_chunks(chunks)

  def save_glossary_to_db(self):
    """Insert short, single-line glossary docs with doc_type='glossary'."""
    docs = []
    for term, definition in GLOSSARY.items():
      content = f"{term}: {definition}"
      docs.append(
        Document(
          page_content=content,
          metadata={
            "region": "Global",
            "doc_type": "glossary",
            "term": term
          }
        )
      )
    if docs:
      db = DB(self.embedding)
      db.insert_chunks(docs)  # Keep it simple; call once to avoid duplicates

if __name__ == "__main__":
  manager = DocumentManager("./regulations/")
  manager.process_documents()
