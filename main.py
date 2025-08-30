# main.py
import argparse, json, os

from pprint import pprint
from retriever_service import RetrieverService
from llm_service import LLMService
from rag_chain import build_rag_chain, extract_json  # the builder we set up for RetrievalQA
from db_orchestrator import DBOrchestrator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from code_change_evaluator import CodeChangeEvaluator
from dev_doc_evaluator import DevDocEvaluator

def process_query(llm, query, k):
    AVAILABLE_REGIONS = [
        "Utah",
        "United States",
        "European Union",
        "California",
        "Florida",
        "Global"
    ]

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

        Query: {query}
        Answer:"""

    response = llm.pipe(prompt)[0]['generated_text']
    # Find first next line
    index = response.find("\n")
    response = response[:index]
    response = response.strip()
    regions = response.split(", ")
    
    # 2) Load retriever and retrievalservice according to regions
    retrievers = db_orchestrator.get_retriever_by_region(regions)
    retrieverServices = RetrieverService(k=k, retriever=retrievers, embedding=embeddings)

    # 3) Build the QA chain (stuff prompt, return sources)
    qa = build_rag_chain(retrieverServices, llm)

    # 4) Get the RAW result (dict) and print it verbatim
    raw = qa.invoke({"query": query})

    # print("\n--- RAW DICT ---")
    # pprint(raw)  # shows everything without any parsing

    obj = extract_json(raw['result'])
    print("\n--- PARSED JSON ---")
    pprint(obj)  # shows the parsed JSON object
    return obj

def evaluate_code_change(llm, json_path):
    code_change_evaluator = CodeChangeEvaluator(llm)
    response = code_change_evaluator.evaluate(json_path)
    print(f'Code change evaluation: {response}')
    return response

def evaluate_dev_doc(llm, dev_doc_dir):
    dev_doc_evaluator = DevDocEvaluator(llm)
    response = dev_doc_evaluator.evaluate(dev_doc_dir)
    print(f'Dev doc evaluation: {response}')
    return response

def main():
    parser = argparse.ArgumentParser(description="Run RetrievalQA and print the RAW result dict.")
    parser.add_argument("-query", "--query", help="User query / feature description to evaluate.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Top-k documents to retrieve (default: 5).")
    
    parser.add_argument("-evaluate_code", "--evaluate_code",type=str, help="Evaluate the code change stored in json path")
    parser.add_argument("-evaluate_doc", "--evaluate_doc",type=str, help="Evaluate the dev doc stored in json path")
    args = parser.parse_args()

    llm = LLMService()

    if args.evaluate_code:
        # Check if the file exists
        if not os.path.exists(args.evaluate_code):
            print(f"Error: {args.evaluate_code} does not exist")
            return

        code_changes = evaluate_code_change(llm, args.evaluate_code)
        # Save code change evaluation into txt file
        # Check if the directory exists
        if not os.path.exists('code_change_eval'):
            os.makedirs('code_change_eval')

        with open('code_change_eval/code_changes.txt', 'w') as f:
            for code_change in code_changes:
                f.write(f'{code_change}\n')

        for code_change in code_changes:
            print(f'code_change: {code_change}')
            query = code_change['feature_name'] + ' ' + code_change['feature_description']
            response = process_query(llm, query, args.k)

            # Save query response into txt file
            if not os.path.exists('code_change_geocompliance'):
                os.makedirs('code_change_geocompliance')
            with open(f'code_change_geocompliance/{code_change["file"]}.txt', 'w') as f:
                f.write(f'{response}\n')

    elif args.evaluate_doc:
        # Check if the file exists
        if not os.path.exists(args.evaluate_doc):
            print(f"Error: {args.evaluate_doc} does not exist")
            return

        dev_docs = evaluate_dev_doc(llm, args.evaluate_doc)
        # Save code change evaluation into txt file
        # Check if the directory exists
        if not os.path.exists('dev_doc_eval'):
            os.makedirs('dev_doc_eval')

        for dev_doc in dev_docs:
            with open(f'dev_doc_eval/{dev_doc["file"]}_features.txt', 'w') as f:
                geocompliance_responses = []
                for feature in dev_doc['features']:
                    f.write(f'{feature["feature_name"]} {feature["feature_description"]}\n')

                    query = feature['feature_name'] + ' ' + feature['feature_description']
                    print(f'query: {query}')
                    try:
                        response = process_query(llm, query, args.k)
                    except Exception as e:
                        print(f'Error processing feature: {feature}: {e}')
                        continue
                    
                    try:
                        geocompliance_responses.append(response)
                    except Exception as e:
                        print(f'Error processing feature: {feature}: {e}')

            # Save query response into txt file
            if not os.path.exists('dev_doc_geocompliance'):
                os.makedirs('dev_doc_geocompliance')
            with open(f'dev_doc_geocompliance/{dev_doc["file"]}_features.txt', 'w') as f:
                f.write(f'{geocompliance_responses}\n')

    elif args.query:
        response = process_query(llm, args.query, args.k)
        print(f'Response: {response}')

if __name__ == "__main__":
    # Run as: python main.py "Your query here" -k 5
    main()