# main.py
import argparse, json
from pprint import pprint
from retriever_service import RetrieverService
from llm_service import LLMService
from rag_chain import build_rag_chain  # the builder we set up for RetrievalQA
from db_orchestrator import DBOrchestrator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def extract_json(raw: str) -> dict:
    # Drop code fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[-1] if parts else raw
    # Take the last {...} block
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("No JSON object found in input.")
    return json.loads(raw[s:e+1])

def main():
    parser = argparse.ArgumentParser(description="Run RetrievalQA and print the RAW result dict.")
    parser.add_argument("query", help="User query / feature description to evaluate.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Top-k documents to retrieve (default: 5).")
    args = parser.parse_args()

    AVAILABLE_REGIONS = [
        "Utah",
        "United States",
        "European Union",
        "California",
        "Florida",
        "Global"
    ]
    llm = LLMService()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # Create the embeddings object once and reuse it
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db_orchestrator = DBOrchestrator(embeddings)

    # Method 1: Few-shot prompting with examples
    prompt = f"""System: Classify this query into geographic regions. Return only the region names and end response.
        Available regions: {', '.join(AVAILABLE_REGIONS)}

        Examples:
        Query: "California privacy law requires..."
        Answer: California

        Query: "Utah social media restrictions for minors..."
        Answer: Utah

        Query: "EU GDPR compliance and US regulations..."
        Answer: European Union, United States

        Query: {args.query}
        Answer:"""

    response = llm.pipe(prompt)[0]['generated_text']
    # Find first next line
    index = response.find("\n")
    response = response[:index]
    response = response.strip()
    regions = response.split(", ")
    
    # 2) Load retriever and retrievalservice according to regions
    retrievers = db_orchestrator.get_retriever_by_region(regions)
    retrieverServices = RetrieverService(k=args.k, retriever=retrievers, embedding=embeddings)

    # 3) Build the QA chain (stuff prompt, return sources)
    qa = build_rag_chain(retrieverServices, llm)

    # 4) Get the RAW result (dict) and print it verbatim
    raw = qa.invoke({"query": args.query})

    # print("\n--- RAW DICT ---")
    # pprint(raw)  # shows everything without any parsing

    obj = extract_json(raw['result'])
    print("\n--- PARSED JSON ---")
    pprint(obj)  # shows the parsed JSON object
if __name__ == "__main__":
    # Run as: python main.py "Your query here" -k 5
    main()