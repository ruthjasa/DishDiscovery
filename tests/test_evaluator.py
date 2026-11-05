import unittest
from src.evaluator import IREvaluator

class MockSearchEngine:
    def search(self, query: str, top_k: int = 5):
        if "squash" in query:
            return [{"id": 137739}, {"id": 31490}]
        elif "breakfast" in query:
            return [{"id": 31490}]
        return []

class TestIREvaluator(unittest.TestCase):
    def test_prec_at_k(self):
        retrieved = [101, 102, 103, 104, 105]
        relevant = {101, 103, 999}

        score = IREvaluator.precision_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(score, 2/3)

    def test_average_prec(self):
        retrieved = [101, 102, 103]
        relevant = {101, 103}

        score = IREvaluator.average_precision(retrieved, relevant)
        self.assertAlmostEqual(score, 5/6)

    def test_ndcg_at_k(self):
        retrieved = [101, 102]
        relevant_scores = {101: 2, 102: 1}

        score = IREvaluator.ndcg_at_k(retrieved, relevant_scores, k=2)
        self.assertAlmostEqual(score, 1.0)

    def test_eval_search_engine(self):
        mock_engine = MockSearchEngine()
        metrics = IREvaluator.evaluate_search_engine(mock_engine, benchmark_file="data/test_queries.json", k=2)
        self.assertIn("MAP", metrics)
        self.assertIn("Mean Precision@2", metrics)

if __name__ == '__main__':
    unittest.main()