"""Dish Discovery: BM25F-style recipe search and metadata filters."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

DIET_TAGS = {
    "vegetarian": {"vegetarian"},
    "vegan": {"vegan"},
    "gluten-free": {"gluten-free"},
    "dairy-free": {"dairy-free"},
}

FIELDS = ("name", "description", "ingredients", "tags")
FIELD_WEIGHTS = {"name": 3.0, "description": 1.0, "ingredients": 2.0, "tags": 1.5}

class RecipeSearch:
    """Kent: Search engine initialized from Teammate A's RecipeIndexer."""

    def __init__(self, indexer, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.indexer = indexer

        self.recipes: dict[int, dict[str, Any]] = {
            int(row["id"]): row.to_dict() for _, row in self.indexer.df.iterrows()
        }

        # Structure: token -> list of (recipe_id, field_name, term_frequency)
        self.postings: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        self.field_lengths: dict[int, dict[str, int]] = {}
        self.avg_lengths: dict[str, float] = {}
        self.doc_freq: dict[str, int] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Constructs the BM25F inverted index using preprocessed data from RecipeIndexer."""
        length_totals = Counter()

        for _, recipe in self.indexer.df.iterrows():
            rid = int(recipe["id"])
            
            field_texts = {
                "name": recipe.get("name", ""),
                "description": recipe.get("description", ""),
                "ingredients": " ".join(recipe.get("ingredients", [])) if isinstance(recipe.get("ingredients"), list) else str(recipe.get("ingredients", "")),
                "tags": " ".join(recipe.get("tags", [])) if isinstance(recipe.get("tags"), list) else str(recipe.get("tags", "")),
            }

            self.field_lengths[rid] = {}
            for field, text in field_texts.items():
                tokens = self.indexer.preprocess_text(text)
                counts = Counter(tokens)
                length = sum(counts.values())

                self.field_lengths[rid][field] = length
                length_totals[field] += length

                for term, tf in counts.items():
                    self.postings[term].append((rid, field, tf))

        n = len(self.recipes)
        self.avg_lengths = {
            f: max(length_totals[f] / n, 1.0) if n else 1.0 for f in FIELDS
        }
        self.doc_freq = {
            term: len({rid for rid, _, _ in entries})
            for term, entries in self.postings.items()
        }

    @staticmethod
    def _matches(
        recipe: dict[str, Any],
        *,
        max_minutes: int | None,
        cuisine: str | None,
        diet: str | None,
        include_ingredients: list[str] | None,
        exclude_ingredients: list[str] | None,
    ) -> bool:
        if max_minutes is not None and recipe["minutes"] > max_minutes:
            return False

        tags = set(recipe.get("tags", []))
        if cuisine and cuisine.strip().lower() not in tags:
            return False

        if diet:
            normalized = diet.strip().lower().replace("_", "-")
            if normalized not in DIET_TAGS:
                raise ValueError(f"Unknown diet {diet!r}; choose {list(DIET_TAGS)}")
            if not (tags & DIET_TAGS[normalized]):
                return False

        ingredient_text = " ".join(recipe.get("ingredients", []))
        if include_ingredients and not all(
            phrase.lower().strip() in ingredient_text for phrase in include_ingredients
        ):
            return False

        if exclude_ingredients and any(
            phrase.lower().strip() in ingredient_text for phrase in exclude_ingredients
        ):
            return False

        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        max_minutes: int | None = None,
        cuisine: str | None = None,
        diet: str | None = None,
        include_ingredients: list[str] | None = None,
        exclude_ingredients: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if max_minutes is not None and max_minutes < 0:
            raise ValueError("max_minutes cannot be negative")

        terms = set(self.indexer.preprocess_text(query))
        n = len(self.recipes)
        scores: dict[int, float] = defaultdict(float)

        for term in terms:
            entries = self.postings.get(term, [])
            if not entries:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

            weighted_tfs: dict[int, float] = defaultdict(float)
            for rid, field, tf in entries:
                length = self.field_lengths[rid][field]
                denom = 1 - self.b + self.b * length / self.avg_lengths[field]
                weighted_tfs[rid] += FIELD_WEIGHTS[field] * tf / denom

            for rid, weighted_tf in weighted_tfs.items():
                scores[rid] += (
                    idf * (self.k1 + 1) * weighted_tf / (self.k1 + weighted_tf)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results = []
        for rid, score in ranked:
            recipe = self.recipes[rid]
            if self._matches(
                recipe,
                max_minutes=max_minutes,
                cuisine=cuisine,
                diet=diet,
                include_ingredients=include_ingredients,
                exclude_ingredients=exclude_ingredients,
            ):
                results.append({**recipe, "score": round(score, 4)})
                if len(results) == top_k:
                    break

        return results