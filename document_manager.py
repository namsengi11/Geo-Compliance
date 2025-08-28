import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from document_loader import DocumentLoader
from db import DB

class DocumentManager():
  def __init__(self, dir):
    self.dir = dir
    self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    self.embedding = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)

  def process_documents(self, chunk_size=1000, chunk_overlap=500):
    '''
    Chunks the documents and saves them to the database as embeddings.
    Takes regional metadata from texts-available.txt.
    Saves the embeddings to the database of each region.
    '''

    # texts-available.txt contains the metadata of the documents
    with open("texts-available.csv", "r") as f:
      # Skip header
      texts_available = f.readline()
      texts_available = f.readlines()
    
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      add_start_index=True
    )

    print(texts_available)
    for text_name in texts_available:
      region, text_name = text_name.split(",")
      try:
        loader = DocumentLoader(os.path.join(self.dir, text_name))
        document = loader.load()
      except Exception as e:
        print(f"Error loading {text_name}: {e}")
        continue

      chunks = text_splitter.split_documents(document)
      self.save_to_db(region, chunks)

  def save_to_db(self, region, chunks):
    db = DB(region, self.embedding)
    db.insert_chunks(chunks)

if __name__ == "__main__":
  manager = DocumentManager("./regulations/")
  manager.process_documents()

