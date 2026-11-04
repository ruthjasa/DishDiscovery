import pandas as pd

class RecipeIndexer:
    """Teammate A: Preprocessing & Inverted Indexing"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.processed_docs = []
        
    def load_and_preprocess(self):
        # TODO: Parse CSV, process ingredients/tags/steps, and build processed_docs
        pass