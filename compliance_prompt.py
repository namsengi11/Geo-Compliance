from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

tok = None  # lazy init

def _to_chat(s: str, model_name: str) -> str:
    global tok
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name)
    msgs = [{"role": "user", "content": s}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def compliance_prompt(model_name: str | None = None):
    base = PromptTemplate.from_template(
        "You are a compliance assistant for geo-regulations.\n"
        "Use ONLY the provided context. If insufficient, say you don't know.\n"
        "Respond in TWO parts:\n"
        "A) Start with 'Answer:' and give 2-4 concise sentences.\n"
        "B) On a new line, output a JSON object exactly like:\n"
        '{{"compliance_needed": <true|false>, "issues": [<strings>], "jurisdictions": [<strings>], '
        '"confidence": <0..1>, "reasoning":"<short>", '
        '"citations":[{{"source":"<path or id>", "start_index":"<number or n/a>"}}]}}\n\n'
        "Question: {question}\n\nContext:\n{context}\n"
    )
    if not model_name:
        return base
    # decorate the formatter to output chat-formatted text - there is a need to set appropriate template for a selected model
    def _format_chat(**kwargs):
        s = base.format(**kwargs)
        return {"question": kwargs["question"], "context": kwargs["context"], "text": _to_chat(s, model_name)}
    # LangChain will pass kwargs; we want final string -> LLM
    base.format = _format_chat  
    return base