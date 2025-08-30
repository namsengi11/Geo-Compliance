import os
import pypdf

from llm_service import LLMService
from evaluate_dev_doc_prompt import evaluate_dev_doc_prompt
from bs4 import BeautifulSoup
from markdown import Markdown
from rag_chain import extract_json

class DevDocEvaluator:
    def __init__(self, llm: LLMService):
        self.llm = llm
        self.EVALUATE_PROMPT = evaluate_dev_doc_prompt()

    def extract_dir_contents(self, dev_doc_dir: str) -> dict:
        # Parse different types of dev docs
        contents = {}
        for file in os.listdir(dev_doc_dir):
            contents[file] = self.extract_contents(os.path.join(dev_doc_dir, file))
        return contents

    def extract_contents(self, dev_doc_path: str) -> dict:
        content = []
        match dev_doc_path.split('.')[-1]:
            case 'pdf':
                with open(dev_doc_path, 'rb') as openFile:
                    pdf_reader = pypdf.PdfReader(openFile)
                    for page in pdf_reader.pages:
                        content.append(page.extract_text())
            case 'html':
                with open(dev_doc_path, 'r', encoding='utf-8') as openFile:
                    html_reader = BeautifulSoup(openFile, 'html.parser')
                    content.append(html_reader.prettify())
            case 'md':
                with open(dev_doc_path, 'r', encoding='utf-8') as openFile:
                    md_reader = Markdown(openFile)
                    content.append(md_reader.content)
            case _:
                print(f'Unsupported file type: {dev_doc_path}')
                content.append('')
        return content

    def evaluate(self, dev_doc_path: str) -> list:
        # If path is directory, extract all contents
        if os.path.isdir(dev_doc_path):
            contents = self.extract_dir_contents(dev_doc_path)
        else:
            fileName = dev_doc_path.split('/')[-1]
            contents = {fileName: self.extract_contents(dev_doc_path)}
        
        responses = []
        for file, content in contents.items():
            prompt = self.EVALUATE_PROMPT.format(context=content)
            response = self.llm.pipe(prompt)[0]['generated_text']
            try:
                responses.append(extract_json(response))
            except Exception as e:
                print(f'Error extracting JSON from {file}: {e}')
                if response[-1] != '}':
                    response = response + '}'
                    try:
                        responses.append(extract_json(response))
                    except Exception as e:
                        print(f'Error extracting JSON from {file}: {e}')
        return responses
    
    def evaluate_doc_from_change(self, change: dict):
        pass

if __name__ == "__main__":
    llm = LLMService()
    dev_doc_evaluator = DevDocEvaluator(llm)
    texts = dev_doc_evaluator.evaluate('dev_docs/example_prd.pdf')
    print(texts)
    # print(texts)