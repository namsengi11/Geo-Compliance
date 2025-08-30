import argparse, json, torch
from pprint import pprint

from retriever_service import RetrieverService
from llm_service import LLMService
from db_orchestrator import DBOrchestrator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from gemini_llm_service import GeminiLLMService

# Use the plain-text RAG call that returns AIMessage metadata
from rag_chain import run_rag_plaintext, build_rag_chain  # build_rag_chain kept for optional comparison

def safe_to_dict(answer):
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, (bytes, bytearray)):
        answer = answer.decode("utf-8", errors="ignore")
    if isinstance(answer, str):
        try:
            return json.loads(answer)
        except Exception:
            return {"_raw": answer}
    return {"_raw": str(answer)}

def main():
    parser = argparse.ArgumentParser(description="Run RAG and print parsed JSON.")
    parser.add_argument("query", help="User query / feature description to evaluate.")
    parser.add_argument("--provider", choices=["hf","gemini"], default="gemini",
                        help="Which LLM backend to use.")
    parser.add_argument("-k", "--k", type=int, default=5,
                        help="Top-k documents to retrieve (default: 5).")
    parser.add_argument("--also-stuff", action="store_true",
                        help="Also run the legacy stuff-docs chain (for comparison).")
    args = parser.parse_args()

    AVAILABLE_REGIONS = ["Utah","United States","European Union","California","Florida","Global"]

    # 1) Pick LLM backend
    if args.provider == "gemini":
        service = GeminiLLMService(model_json="gemini-2.5-flash", model_text="gemini-2.5-flash", max_output_tokens=1536)
    else:
        service = LLMService()
    llm = service.llm  # <— THIS is the object we pass to run_rag_plaintext

    # 2) Embeddings (CUDA if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3) Vector DB + retriever(s)
    db_orchestrator = DBOrchestrator(embeddings)

    # (Optional) simple region classifier — disabled for now; hardcode regions:
    # prompt = f"""System: Classify this query into geographic regions. Return only the region names and end response.
    #     Available regions: {', '.join(AVAILABLE_REGIONS)}
    #     Examples:
    #     Query: "California privacy law requires..."   -> California
    #     Query: "Utah social media restrictions..."   -> Utah
    #     Query: "EU GDPR + US regs..."                -> European Union, United States
    #     Query: {args.query}
    #     Answer:"""
    # response = service.generate_text(prompt).splitlines()[0].strip()
    # regions = [r.strip() for r in response.split(",") if r.strip()]
    regions = ["Utah", "United States"]  # hardcoded for now

    # Orchestrate region-scoped retrievers and wrap with our RetrieverService
    retrievers = db_orchestrator.get_retriever_by_region(regions)
    retriever_service = RetrieverService(k=args.k, retriever=retrievers, embedding=embeddings)

    # 4) Run the plain-text RAG call that ALWAYS returns something and exposes metadata
    out_plain = run_rag_plaintext(llm=llm, retriever=retriever_service, user_input=args.query)

    # Human-readable answer (never empty; falls back to "[Empty response from model]")
    print(out_plain["answer"])

    # If Gemini adds diagnostics (e.g., safety), show them:
    if out_plain.get("response_metadata"):
        print("\n[DEBUG] response_metadata:")
        pprint(out_plain["response_metadata"])

    # 5) Optional: also run the legacy stuff-docs chain for comparison
    if args.also_stuff:
        rag_chain = build_rag_chain(retriever_service, service)  # legacy string-only chain
        out = rag_chain.invoke({"input": args.query})
        print("\n\n--- Legacy stuff-docs output ---")
        pprint(out)
        answer = out.get("answer", "")
        print("\n[Legacy answer string]:")
        print(answer)
        print("\n--- PARSED JSON (legacy) ---")
        pprint(safe_to_dict(answer))

if __name__ == "__main__":
    main()
