
import ast
import re

import pandas as pd
from nltk.stem import PorterStemmer


class RecipeIndexer:
    """Loads and preprocesses recipe data for BM25 indexing."""

    def __init__(self, data_path: str, ingredient_weight: int = 3):
        self.data_path = data_path
        self.df = None
        self.processed_docs = []
        self.stemmer = PorterStemmer()

        if ingredient_weight < 1:
            raise ValueError("ingredient_weight must be at least 1")

        self.ingredient_weight = ingredient_weight

        # Small built-in stop-word set to avoid downloading NLTK data.
        self.stop_words = {
            "a", "an", "the", "and", "or", "of", "to",
            "in", "on", "for", "with", "at", "by",
            "is", "are", "be", "as", "from"
        }

    @staticmethod
    def parse_list(value):
        """Convert a stringified Python list into a native list."""
        if isinstance(value, list):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        else:
            return []

        if not isinstance(parsed, list):
            return []

        return [item.strip() for item in parsed
                if isinstance(item, str) and item.strip()]

    def preprocess_text(self, text: str) -> list[str]:
        """Normalize text, remove stop words, and stem tokens."""
        if not isinstance(text, str):
            return []

        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)

        return [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words
        ]

    def build_processed_docs(self):
        """Create weighted token lists for BM25 indexing."""
        self.processed_docs = []

        for _, recipe in self.df.iterrows():
            title_tokens = self.preprocess_text(recipe["name"])

            ingredient_tokens = []
            for ingredient in recipe["ingredients"]:
                ingredient_tokens.extend(
                    self.preprocess_text(ingredient)
                )

            document = (
                title_tokens
                + ingredient_tokens * self.ingredient_weight
            )

            self.processed_docs.append(document)

    def load_and_preprocess(self):
        """Load CSV, clean recipe records, and prepare BM25 documents."""
        df = pd.read_csv(self.data_path)

        required_columns = [
            "id", "name", "minutes",
            "ingredients", "tags", "steps"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Keep only the fields required by the shared contract.
        df = df[required_columns].copy()

        # Convert numeric fields and discard invalid records.
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df["minutes"] = pd.to_numeric(
            df["minutes"], errors="coerce"
        )

        df = df.dropna(subset=["id", "name", "minutes"])
        df = df[
            (df["id"] % 1 == 0)
            & (df["minutes"] % 1 == 0)
            & (df["minutes"] >= 0)
            & (df["id"] >= 0)
        ].copy()

        df["id"] = df["id"].astype(int)
        df["minutes"] = df["minutes"].astype(int)

        df["name"] = df["name"].astype(str).str.strip()
        df = df[df["name"] != ""].copy()
        df = df.drop_duplicates(subset=["id"])

        # Parse stringified lists.
        for column in ["ingredients", "tags", "steps"]:
            df[column] = df[column].apply(self.parse_list)

        # Recipes without usable ingredients cannot be indexed well.
        df = df[df["ingredients"].map(bool)].copy()

        # Reset indices to align DataFrame rows with BM25 scores.
        self.df = df.reset_index(drop=True)

        self.build_processed_docs()

        return self.df
