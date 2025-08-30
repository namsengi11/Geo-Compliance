# demo_app.py
import os
import json
import logging
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd

# --- Silence warnings / noisy logs for a clean demo ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for name in [
    "langchain", "langchain_core", "langchain_community",
    "langchain_google_genai", "httpx", "chromadb", "urllib3", "requests",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

# --- Your project imports ---
from gemini_llm_service import GeminiLLMService
from main import process_query  # your RetrievalQA + parsing

# ---------- Helpers: CSV history ----------
HISTORY_COLUMNS = ["timestamp", "feature", "feature_description", "response_json"]

def _ensure_history_df(path: str) -> pd.DataFrame:
    """Load history CSV, tolerating missing headers; return a normalized DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(path)
        # If someone wrote without headers (old script), try to coerce:
        if set(df.columns) & set(HISTORY_COLUMNS) != set(HISTORY_COLUMNS):
            df = pd.read_csv(path, header=None, names=["feature", "feature_description", "response_json"])
            # If no timestamp, inject an approximate one
            df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = df[HISTORY_COLUMNS]
    except Exception:
        # Fallback: try no-header mode
        df = pd.read_csv(path, header=None, names=["feature", "feature_description", "response_json"])
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df[HISTORY_COLUMNS]

    # Fill NaNs
    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[HISTORY_COLUMNS]


def upsert_history(path: str, feature: str, feature_description: str, response_obj: dict) -> None:
    """Upsert a row keyed by 'feature'. Always writes canonical header."""
    df = _ensure_history_df(path)

    # Compact JSON for CSV storage
    resp_str = json.dumps(response_obj, ensure_ascii=False, separators=(",", ":"))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if (df["feature"] == feature).any():
        mask = df["feature"] == feature
        df.loc[mask, "feature_description"] = feature_description
        df.loc[mask, "response_json"] = resp_str
        df.loc[mask, "timestamp"] = ts
    else:
        df.loc[len(df)] = [ts, feature, feature_description, resp_str]

    df.to_csv(path, index=False)


def load_history(path: str) -> pd.DataFrame:
    """Read history CSV and return newest-first."""
    df = _ensure_history_df(path)
    # Sort by timestamp if present
    if "timestamp" in df.columns:
        try:
            df["ts_sort"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("ts_sort", ascending=False).drop(columns=["ts_sort"])
        except Exception:
            pass
    return df


# ---------- UI THEME / STYLES ----------
st.set_page_config(page_title="Geo-Reg Compliance Demo", page_icon="🧭", layout="wide")
st.markdown("""
<style>
:root { --card-bg: #ffffff; --card-border: #e6e6e6; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }

.demo-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 20px 22px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.04);
}

.small-label { font-size: 0.85rem; color: #666; margin-bottom: 0.3rem; }
code { white-space: pre-wrap; }

/* Scrollable history panel */
.history-scroll {
  max-height: 420px;
  overflow-y: auto;
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 8px 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
colA, colB = st.columns([0.65, 0.35])
with colA:
    st.markdown("## 🧭 Geo-Reg Compliance Demo")
    st.markdown(
        "Evaluate a feature description against region-specific regulations using your RAG pipeline + Gemini. "
        "Returns a strict JSON schema, with a persistent CSV log and a collapsible history panel."
    )

with colB:
    with st.container():
        st.markdown("<div class='demo-card'>", unsafe_allow_html=True)
        st.markdown("**Status**")
        key_ok = "GEMINI_API_KEY" in os.environ and bool(os.environ["GEMINI_API_KEY"].strip())
        st.metric("Gemini API key", "Found" if key_ok else "Missing", delta=None)
        st.caption("Set `GEMINI_API_KEY` in your environment before running Streamlit.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- SIDEBAR CONTROLS ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_name = st.selectbox("Gemini model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    k = st.slider("Top-K retrieval", 1, 8, 5)
    max_tokens = st.slider("Max output tokens", 256, 4096, 1536, step=128)
    st.markdown("---")
    # CSV path defaults to the batch logger's file for consistency
    history_csv = st.text_input("Results CSV path", value="sample_data_response.csv",
                                help="Each run upserts by 'feature' into this CSV.")
    st.caption(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------- INPUTS ----------
st.markdown("### Describe Your Feature")
with st.container():
    st.markdown("<div class='demo-card'>", unsafe_allow_html=True)
    feature = st.text_input(
        "Feature name",
        key="feature_name",
        placeholder="e.g., EU_Video_Replies_Pilot"
    )
    feature_desc = st.text_area(
        "Feature description (used for retrieval & analysis)",
        height=180,
        placeholder=(
            "e.g., Trial run of video replies in EEA only. GH manages exposure; "
            "BB is used to baseline feedback. No personalization changes planned."
        ),
    )
    feature_desc = feature + ": " + feature_desc 
    st.markdown("</div>", unsafe_allow_html=True)

run = st.button("▶️ Evaluate compliance", use_container_width=True)

# ---------- RUN PIPELINE ----------
if run:
    if not feature_desc.strip():
        st.error("Please provide a feature description.")
    elif not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Please export it in your environment and restart.")
    else:
        with st.spinner("Running retrieval + Gemini…"):
            # Instantiate Gemini service
            service = GeminiLLMService(
                model_json=model_name,
                model_text=model_name,
                max_output_tokens=max_tokens,
            )

            try:
                # Your existing pipeline (returns parsed JSON/dict)
                # Note: We do not concatenate feature+desc for the LLM unless that's your intended input design.
                result_obj = process_query(service, feature_desc, k)

                # ---------- DISPLAY ----------
                st.success("Evaluation complete.")

                # Markdown code view (no fences inside the string)
                st.markdown("#### JSON (Markdown)")
                st.markdown(result_obj)

                # Download button
                st.download_button(
                    label="Download JSON",
                    data=result_obj,
                    file_name=f"{(feature or 'response').strip()}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                # ---------- CSV UPSERT ----------
                if not feature.strip():
                    st.info("Tip: Provide a 'Feature name' to store this run in the CSV history.")
                else:
                    upsert_history(history_csv, feature.strip(), feature_desc.strip(), result_obj)
                    st.success(f"Saved/updated **{feature.strip()}** in `{history_csv}`.")

            except Exception as e:
                st.error(f"Error while processing: {e}")

# ---------- HISTORY PANEL ----------
st.markdown("### History")
with st.expander("Show past queries & responses", expanded=False):
    df = load_history(history_csv)
    if df.empty:
        st.caption("No history yet. Runs will be added here.")
    else:
        # Optional quick table
        st.dataframe(
            df[["timestamp", "feature", "feature_description"]],
            use_container_width=True,
            hide_index=True
        )
        # Scrollable detailed view
        st.markdown("<div class='history-scroll'>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            st.markdown(f"**{row['feature']}** — _{row['timestamp']}_")
            st.markdown(f"> {row['feature_description']}")
            try:
                obj = json.loads(row["response_json"])
                st.code(row["response_json"])
            except Exception:
                # If old rows were stored as raw strings without proper JSON
                st.code(str(row["response_json"]), language="json")
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)

# End of file
