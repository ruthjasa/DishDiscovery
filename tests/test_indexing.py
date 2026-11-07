
import unittest
import os
import tempfile

import pandas as pd

from src.indexer import RecipeIndexer


class TestRecipeIndexer(unittest.TestCase):

    def setUp(self):
        self.indexer = RecipeIndexer("data/sample_recipes.csv")

    def test_parse_list(self):
        result = self.indexer.parse_list(
            "['winter squash', 'butter', 'salt']"
        )
        self.assertEqual(
            result, ["winter squash", "butter", "salt"]
        )

    def test_invalid_list(self):
        self.assertEqual(
            self.indexer.parse_list("not a list"), []
        )
        self.assertEqual(
            self.indexer.parse_list(None), []
        )

    def test_preprocess_text(self):
        result = self.indexer.preprocess_text(
            "Baked Vegetables with Butter!"
        )
        self.assertEqual(
            result, ["bake", "veget", "butter"]
        )

    def test_ingredient_weighting(self):
        self.indexer.load_and_preprocess()

        tokens = self.indexer.processed_docs[0]

        # Butter appears once in the ingredient list,
        # so its weighted frequency should be three.
        self.assertEqual(tokens.count("butter"), 3)

    def test_dataframe_alignment(self):
        self.indexer.load_and_preprocess()

        self.assertEqual(
            len(self.indexer.df),
            len(self.indexer.processed_docs)
        )

    def test_parsed_columns(self):
        self.indexer.load_and_preprocess()

        for column in ["ingredients", "tags", "steps"]:
            self.assertTrue(
                self.indexer.df[column].map(
                    lambda value: isinstance(value, list)
                ).all()
            )

    def test_invalid_rows(self):
        test_data = pd.DataFrame([
            {
                "id": 1,
                "name": "Valid Recipe",
                "minutes": 30,
                "ingredients": "['butter']",
                "tags": "['easy']",
                "steps": "['mix ingredients']"
            },
            {
                "id": 2,
                "name": "Invalid Recipe",
                "minutes": None,
                "ingredients": "['salt']",
                "tags": "[]",
                "steps": "[]"
            }
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "recipes.csv")
            test_data.to_csv(path, index=False)

            indexer = RecipeIndexer(path)
            indexer.load_and_preprocess()

            self.assertEqual(len(indexer.df), 1)
            self.assertEqual(indexer.df.iloc[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
