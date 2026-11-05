import numpy as np
import json

class IREvaluator:
    """Ruth: IR Metrics (Precision@K, MAP, NDCG)"""
    @staticmethod
    def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
        # TODO: Calculate Precision@K
        if not retrieved_ids or k <= 0:
            return 0.0

        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_ids]

        return len(relevant_retrieved)/k

    @staticmethod
    def average_precision(retrieved_ids: list, relevant_ids: set) -> float:
        # calculate average precision for a single query
        if not relevant_ids or not retrieved_ids:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                hits += 1
                prec_at_i = hits / (i+1)
                sum_precisions += prec_at_i

        return sum_precisions / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def mean_average_precision(all_retrieved_ids: list[list], all_relevant_ids: list[set]) -> float:
        # TODO: Calculate MAP
        if not all_retrieved_ids or not all_relevant_ids:
            return 0.0

        ap_scores = [
            IREvaluator.average_precision(ret, rel) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)
        ]
        return float(np.mean(ap_scores) / len(ap_scores)) if ap_scores else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list, relevant_scores: dict, k: int = 10) -> float:
        # TODO: Calculate NDCG@K
        if not retrieved_ids or k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]

        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            rel = relevant_scores.get(str(doc_id), relevant_scores.get(doc_id, 0))
            if rel > 0:
                dcg += rel / (1.0 if i == 0 else (i+1))

        ideal_rel = sorted(relevant_scores.values(), reverse=True)[:k]
        idcg = sum(rel / (1.0 if i == 0 else (i+1)) for i, rel in enumerate(ideal_rel))

        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def evaluate_search_engine(search_engine, benchmark_file: str = "data/test_queries.json", k: int = 5) -> dict:
        with open(benchmark_file, "r") as f:
            test_cases = json.load(f)

        all_retrieved = []
        all_relevant = []
        precisions = []
        ndcgs = []

        for case in test_cases:
            query = case["query_text"]
            relevant_ids = set(case["relevant_recipe_ids"])
            relevance_grades = case.get("relevance_grades", {})

            results = search_engine.search(query, top_k=k)
            
            # Extract IDs whether results is a DataFrame, list of dicts, or list of IDs
            if hasattr(results, "empty") and not results.empty:
                retrieved_ids = results["id"].tolist()
            elif isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict):
                    retrieved_ids = [r["id"] for r in results]
                else:
                    retrieved_ids = results
            else:
                retrieved_ids = []

            p_at_k = IREvaluator.precision_at_k(retrieved_ids, relevant_ids, k=k)
            ndcg = IREvaluator.ndcg_at_k(retrieved_ids, relevance_grades, k=k)

            precisions.append(p_at_k)
            ndcgs.append(ndcg)
            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_ids)

        map_score = IREvaluator.mean_average_precision(all_retrieved, all_relevant)
        mean_p_at_k = float(sum(precisions) / len(precisions)) if precisions else 0.0
        mean_ndcg = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0

        return {
            f"Mean Precision@{k}": round(mean_p_at_k, 4),
            "MAP": round(map_score, 4),
            f"Mean NDCG@{k}": round(mean_ndcg, 4)
        }