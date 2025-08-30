import json

from llm_service import LLMService
from evaluate_change_prompt import evaluate_change_prompt
from rag_chain import extract_json

class CodeChangeEvaluator:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def evaluate(self, json_path: str):
        try: 
            with open(json_path, "r", encoding='utf-8') as f:
                # JSON has attributes: changed_file: {file_path, line_changes: {line_number, change}}
                # change: type, optional(previous_line), optional(new_line)
                # type: "add", "remove", "modify"
                evaluate = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            raise Exception(f"Error loading {json_path}: {e}")
        
        EVALUATE_PROMPT = evaluate_change_prompt()

        # Package change and entire file
        evaluated_changes = []
        for changed_file in evaluate['changed_files']:
            file_path = changed_file['file_path']
            with open(file_path, "r", encoding='utf-8') as f:
                file_content = f.read()

            # Populate prompt with file_path and file_change
            prompt = EVALUATE_PROMPT.format(context=file_path, change=file_content)

            response = self.llm.pipe(prompt)[0]['generated_text']
            obj = extract_json(response)
            evaluated_changes.append(obj)

        return evaluated_changes