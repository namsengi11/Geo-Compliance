# terminology.py
import re
from typing import Dict, List, Tuple

GLOSSARY: Dict[str, str] = {
    "NR": "Not recommended",
    "PF": "Personalized feed",
    "GH": "Geo-handler; a module responsible for routing features based on user region",
    "CDS": "Compliance Detection System",
    "DRT": "Data retention threshold; duration for which logs can be stored",
    "LCP": "Local compliance policy",
    "Redline": "Flag for legal review (different from its traditional business use for 'financial loss')",
    "Softblock": "A user-level limitation applied silently without notifications",
    "Spanner": "A synthetic name for a rule engine (not to be confused with Google Spanner)",
    "ShadowMode": "Deploy feature in non-user-impact way to collect analytics only",
    "T5": "Tier 5 sensitivity data; more critical than T1–T4 in this internal taxonomy",
    "ASL": "Age-sensitive logic",
    "Glow": "A compliance-flagging status, internally used to indicate geo-based alerts",
    "NSP": "Non-shareable policy (content should not be shared externally)",
    "Jellybean": "Feature name for internal parental control system",
    "EchoTrace": "Log tracing mode to verify compliance routing",
    "BB": "Baseline Behavior; standard user behavior used for anomaly detection",
    "Snowcap": "A synthetic codename for the child safety policy framework",
    "FR": "Feature rollout status",
    "IMT": "Internal monitoring trigger",
}

# Build regex once for term detection (whole words, case-sensitive for acronyms/codenames)
_TERMS = sorted(GLOSSARY.keys(), key=len, reverse=True)
_TERM_RE = re.compile(r"\b(" + "|".join(map(re.escape, _TERMS)) + r")\b")

def detect_terms(text: str) -> List[str]:
    if not text:
        return []
    return sorted(set(m.group(1) for m in _TERM_RE.finditer(text)))

def expand_query(query: str) -> str:
    """Append meaning next to terms to help retrieval, keeping the original phrasing."""
    if not query:
        return ""
    def repl(m):
        t = m.group(1)
        expl = GLOSSARY.get(t, "")
        return f"{t} ({expl})" if expl else t
    return _TERM_RE.sub(repl, query)

def make_definitions_block(query: str, max_items: int = 10) -> str:
    """Make a compact block to inject into the prompt. Do NOT treat as legal evidence."""
    hits = detect_terms(query)
    if not hits:
        return "None."
    pairs: List[Tuple[str, str]] = [(t, GLOSSARY[t]) for t in hits][:max_items]
    lines = [f"- {t}: {exp}" for t, exp in pairs]
    return "\n".join(lines)
