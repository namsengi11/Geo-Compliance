import os
from typing import Union
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from db import DB

class DBOrchestrator:
    CHROMA_BASE_PATH = "chroma/"

    def __init__(self, embedding: HuggingFaceEmbeddings):
        self.embedding = embedding
        regions = os.listdir(self.CHROMA_BASE_PATH)
        self.db_by_region = {}
        for region in regions:
            self.db_by_region[region] = DB(region, self.embedding)

    def get_retriever_by_region(self, region):
        '''
        Get the retriever for a given region

        Parameters:
            region: str or list of str

        Output:
            dict: key: region, value: retriever
        '''
        if isinstance(region, str):
            return self.db_by_region[region].get_retriever()

        # Input is list
        retrievers = {}
        try:
            for r in region:
                retrievers[r] = self.get_retriever_by_region(r)
        except KeyError:
            print(f"Region {r} not found in the database")
        return retrievers
            