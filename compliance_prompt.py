# compliance_prompt.py
from langchain.prompts import PromptTemplate

def compliance_prompt(max_issues: int = 5, constraints: str = "AllowedCitations: []") -> PromptTemplate:
    _SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    _USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    _ASSIST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    SYSTEM = (
        "You are a geo-regulation compliance assistant.\n"
        "Use ONLY the provided Context (statutes/regulations) and the user's feature description.\n"
        "Ignore implementation nicknames unless they appear in Context.\n\n"
        "OUTPUT MODE (JSON-ONLY)\n"
        "• Return a SINGLE valid JSON object and nothing else (no prose, no code fences).\n"
        "• Keys and types (exactly these, no extras):\n"
        '  - "compliance_needed": boolean\n'
        f'  - "issues": string[]   // concrete gaps vs the law; max {max_issues} items; each ≤ 20 words; start with "Missing", "Unclear", or "Must"\n'
        '  - "reasoning": string  // 1–2 sentences referencing law anchors; do NOT restate the feature\n'
        '  - "jurisdiction": string[]  // jurisdictions explicitly named in Context (e.g., "Utah")\n'
        '  - "citation": string[] // ≤ 5 UNIQUE law anchors used, e.g., "13-63-105(1)"\n\n'
        "DECISION RULES\n"
        "• If ANY requirement is unmet or unclear ⇒ compliance_needed=true.\n"
        "• Each issue MUST be grounded in Context and supported by ≥1 anchor listed in citation.\n"
        "• citation MUST be ≤5 unique items; choose ONLY from AllowedCitations if provided; do NOT output generic labels like 'Utah Code'.\n"
        "• Keep output concise; avoid long quotes.\n\n"
        "FALLBACK WHEN INSUFFICIENT CONTEXT\n"
        "• If Context lacks the requirements to decide, output exactly:\n"
        "{% raw %}{\"compliance_needed\": false, \"issues\": [], \"reasoning\": \"insufficient context to determine compliance\", \"jurisdiction\": [], \"citation\": []}{% endraw %}\n\n"
        "EMIT JSON WITH THIS SHAPE:\n"
        "{% raw %}{\"compliance_needed\": <true|false>, \"issues\": [\"...\"], \"reasoning\": \"...\", \"jurisdiction\": [\"...\"], \"citation\": [\"13-63-105(1)\"]}{% endraw %}\n"
    )

    USER = (
        "Feature description:\n{{ question }}\n\n"
        "Constraints (may be empty; pick citations only from list if present):\n{{ constraints }}\n\n"
        "Context (authoritative; quote only short snippets ≤ 30 words if needed):\n{{ context }}\n"
    )

    template = _SYS + SYSTEM + _USER + USER + _ASSIST
    return PromptTemplate(
        template=template,
        template_format="jinja2",
        input_variables=["question", "context"],
        partial_variables={"constraints": constraints}
    )
