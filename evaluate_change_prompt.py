# compliance_prompt.py
from langchain.prompts import PromptTemplate

def evaluate_change_prompt() -> PromptTemplate:
    _SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    _USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    _ASSIST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    SYSTEM = (
        "You are a code change evaluator.\n"
        "Use ONLY the provided context.\n\n"
        "OUTPUT MODE (JSON-ONLY)\n"
        "• Return a SINGLE valid JSON object and nothing else (no prose, no code fences).\n"
        "• Keys and types (exactly these, no extras):\n"
        '  - "file": string\n'
        '  - "feature_name": string\n // Briefly describe the feature affected by the code change\n\n'
        '  - "feature_description": string\n // Describe the effect achieved by the feature and the purpose of the feature implementation\n\n'
        
        "FALLBACK WHEN INSUFFICIENT CONTEXT\n"
        "• If Context lacks the substantial code change or only has formatting changes:\n"
        "{% raw %}{\"file\": {file name in input json}, \"feature_name\": \"No feature-level change\", \"feature_description\": \"No feature-level change\"}{% endraw %}\n\n"
        "EMIT JSON WITH THIS SHAPE:\n"
        "{% raw %}{\"file\": {file name in input json}, \"feature_name\": \"...\", \"feature_description\": \"...\"}{% endraw %}\n"
    )

    USER = (
        "File Content:\n{{ context }}\n\n"
        "Change:\n{{ change }}\n\n"
        )

    template = _SYS + SYSTEM + _USER + USER + _ASSIST
    return PromptTemplate(
        template=template,
        template_format="jinja2",
        input_variables=["context", "change"],
    )
