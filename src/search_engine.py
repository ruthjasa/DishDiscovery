import pandas as pd

class DishSearchEngine:
    """Teammate B: BM25 Search Engine & Filtering"""
    def __init__(self, indexer=None):
        self.indexer = indexer

    def search(self, query: str, max_minutes: int = None, tag_filter: str = None, top_k: int = 10):
        # TODO: Implement BM25 ranking and filtering on minutes and tags
        pass