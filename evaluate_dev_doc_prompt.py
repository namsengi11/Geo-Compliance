# compliance_prompt.py
from langchain.prompts import PromptTemplate

def evaluate_dev_doc_prompt() -> PromptTemplate:
    _SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    _USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    _ASSIST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    SYSTEM = (
        "You are a developer document evaluator.\n"
        "Use ONLY the provided context.\n\n"
        "Focus on how users experience the product and how information is handled by the product.\n"
        "Ignore internal development plans or implementation details that have no relevance to geo-regulation compliance.\n"
        "ENSURE CORRECT JSON OUTPUT FORMAT\n"
        "OUTPUT MODE (JSON-ONLY)\n"
        "• Return a SINGLE valid JSON object and nothing else (no prose, no code fences).\n"
        "• Keys and types (exactly these, no extras):\n"
        '  - "file": string\n'
        '  - "features": list of json object\n // Features described in the developer document\n\n'
        '  - "features[i].feature_name": string\n // Name of the feature\n\n'
        '  - "features[i].feature_description": string\n // Description of the feature and its purpose on users\n\n'
        
        "FALLBACK WHEN INSUFFICIENT CONTEXT\n"
        "• If Context lacks the substantial features:\n"
        "{% raw %}{\"file\": {file name in input json}, \"features\": []}{% endraw %}\n\n"
        "EMIT JSON WITH THIS SHAPE:\n"
        "{% raw %}{\"file\": {file name in input json}, \"features\": [{\"feature_name\": \"...\", \"feature_description\": \"...\"}, ...]}{% endraw %}\n"
    )

    USER = (
        "Developer Document Content:\n{{ context }}\n\n"
        )

    template = _SYS + SYSTEM + _USER + USER + _ASSIST
    return PromptTemplate(
        template=template,
        template_format="jinja2",
        input_variables=["context"],
    )
