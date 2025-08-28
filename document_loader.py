from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
  def __init__(self, file_path):
    file_path = file_path.strip()
    if file_path.endswith(".html"):
      self.loader = BSHTMLLoader(file_path=file_path, open_encoding="utf-8")
    elif file_path.endswith(".pdf"):
      self.loader = PyPDFLoader(file_path=file_path)
    else:
      print(file_path.endswith(".html"), file_path[-5:])
      raise ValueError(f"Unsupported file type: {file_path}")

  def load(self):
    return self.loader.load()