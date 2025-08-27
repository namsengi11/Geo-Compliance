# main.py
import argparse
from pprint import pprint
from retriever_service import RetrieverService
from llm_service import LLMService
from rag_chain import build_rag_chain  # the builder we set up for RetrievalQA

def main():
    parser = argparse.ArgumentParser(description="Run RetrievalQA and print the RAW result dict.")
    parser.add_argument("query", help="User query / feature description to evaluate.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Top-k documents to retrieve (default: 5).")
    args = parser.parse_args()

    # 1) Load retriever (Chroma + MiniLM) and generator (Currently Gemma-3-1b-it)
    retr = RetrieverService(device="cpu", k=args.k)  
    llm = LLMService()  # swap model/device in LLMService if desired with model_name

    # 2) Build the QA chain (stuff prompt, return sources)
    qa = build_rag_chain(retr, llm)

    # 3) Get the RAW result (dict) and print it verbatim
    raw = qa.invoke({"query": args.query})

    print("\n--- RAW DICT ---")
    pprint(raw)  # shows everything without any parsing

if __name__ == "__main__":
    # Run as: python main.py "Your query here" -k 5
    main()