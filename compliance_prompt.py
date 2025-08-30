# compliance_prompt.py
from langchain_core.prompts import ChatPromptTemplate

def compliance_prompt(json_mode: str = "schema", max_words: int = 160):
    """
    json_mode:
      - "schema": default for Gemini plain text (concise bullets; no forced JSON).
      - "text":   for local models where you want strict JSON output.
    max_words: soft cap the model's answer length (helps avoid MAX_TOKENS).
    """
    SYSTEM_CONCISE = (
        "You are a geo-regulation compliance assistant.\n"
        "Rely ONLY on the provided Context; do not add external knowledge.\n"
        "Your output MUST be terse and structured. No preamble. No conclusions.\n"
        f"Hard brevity rule: keep the entire response ≤ {max_words} words.\n"
        "Formatting:\n"
        "• Decision: one of [Compliant, Partially compliant, Unknown]\n"
        "• Why: 2–4 bullets, each ≤ 14 words, cite section numbers if present (e.g., §13-63-105(1)).\n"
        "• Required controls: 1–3 bullets that the statute explicitly requires; if none, write “None in Context”.\n"
        "• Not specified: 1–4 bullets listing feature elements not in Context (e.g., ASL, GH, tools).\n"
        "Rules:\n"
        "1) Do not invent facts/laws not in Context.\n"
        "2) Prefer the most specific subsection (e.g., §13-63-105(3)(a) over §13-63-105(3)).\n"
        "3) If Context is insufficient overall, set Decision to “Unknown” and list why.\n"
        "4) Avoid redundant wording and qualifiers; use compact bullets.\n"
        "End the response with <END>."
    )

    SYSTEM_JSON = (
        "You are a geo-regulation compliance assistant. Use ONLY the provided Context.\n"
        "Output ONLY a single valid JSON object with exactly these keys:\n"
        '  "compliance_needed" (bool),\n'
        '  "issues" (string[] ≤ 5, each ≤ 18 words),\n'
        '  "reasoning" (string ≤ 120 words, concise),\n'
        '  "jurisdiction" (string[]).\n'
        "Do not invent facts; prefer specific subsections (e.g., §13-63-105(3)(a))."
    )

    system = SYSTEM_JSON if json_mode == "text" else SYSTEM_CONCISE

    user = (
        "Feature description:\n{input}\n\n"
        "Context (authoritative excerpts):\n{context}\n"
        "\nReturn only the formatted answer described above. <END>"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", user),
    ])
