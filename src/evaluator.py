class IREvaluator:
    """Teammate C: IR Metrics (Precision@K, MAP, NDCG)"""
    @staticmethod
    def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
        # TODO: Calculate Precision@K
        return 0.0

    @staticmethod
    def mean_average_precision(retrieved_ids: list, relevant_ids: set) -> float:
        # TODO: Calculate MAP
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list, relevant_scores: dict, k: int = 10) -> float:
        # TODO: Calculate NDCG@K
        return 0.0