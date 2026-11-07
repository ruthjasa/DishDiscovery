import csv
import tempfile
import unittest
from pathlib import Path

from src.indexer import RecipeIndexer
from src.search_engine import RecipeSearch


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "recipes.csv"
        recipes = [
            {
                "id": 1,
                "name": "Garlic pasta",
                "description": "quick noodles",
                "ingredients": "['pasta','garlic','olive oil']",
                "tags": "['italian','vegan']",
                "minutes": 20,
                "steps": "['boil pasta', 'add garlic']",
            },
            {
                "id": 2,
                "name": "Creamy pasta",
                "description": "garlic cheese pasta",
                "ingredients": "['pasta','garlic','milk']",
                "tags": "['italian','vegetarian']",
                "minutes": 40,
                "steps": "['cook noodles']",
            },
            {
                "id": 3,
                "name": "Garlic soup",
                "description": "warming bowl",
                "ingredients": "['garlic','stock']",
                "tags": "['soup','vegan']",
                "minutes": 15,
                "steps": "['simmer garlic']",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=recipes[0].keys())
            writer.writeheader()
            writer.writerows(recipes)

        self.indexer = RecipeIndexer(str(path))
        self.indexer.load_and_preprocess()

        self.engine = RecipeSearch(self.indexer)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bm25_returns_matching_documents(self):
        result = self.engine.search("pasta")
        self.assertEqual({r["id"] for r in result}, {1, 2})
        self.assertGreater(result[0]["score"], 0)

    def test_combined_filters(self):
        result = self.engine.search(
            "garlic", max_minutes=25, cuisine="italian", diet="vegan"
        )
        self.assertEqual([r["id"] for r in result], [1])

    def test_ingredient_filters(self):
        result = self.engine.search(
            "pasta", include_ingredients=["garlic"], exclude_ingredients=["milk"]
        )
        self.assertEqual([r["id"] for r in result], [1])

    def test_empty_query_and_bad_diet(self):
        self.assertEqual(self.engine.search(""), [])
        with self.assertRaises(ValueError):
            self.engine.search("pasta", diet="keto")

    def test_parsing(self):
        self.assertEqual(
            RecipeIndexer.parse_list("['garlic', 'oil']"), ["garlic", "oil"]
        )

if __name__ == "__main__":
    unittest.main()