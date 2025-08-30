import pandas as pd
from gemini_llm_service import GeminiLLMService
import dotenv
dotenv.load_dotenv()
llm = GeminiLLMService()


def process_query(query):
    from main import process_query
    return process_query(llm, query, 5, "gemini")

sample_data = pd.read_csv("sample_data.csv")
# print(sample_data['feature_name'])

with open("sample_data_response.csv", "w") as f:
    for index, line in sample_data.iterrows():
        try:
            feature_name = sample_data.iloc[index]["feature_name"]
            feature_description = sample_data.iloc[index]["feature_description"]
            query = feature_name + " " + feature_description
            response = process_query(query)
            f.write(f"{feature_name},{feature_description},{response}\n")
        except Exception as e:
            print(f"Error processing query: {e}")
            continue

# sample_data['response'] = sample_data.apply(lambda x: process_query(x["feature_name"] + " " + x["feature_description"]), axis=1)
# sample_data.to_csv("sample_data_response.csv", index=False)