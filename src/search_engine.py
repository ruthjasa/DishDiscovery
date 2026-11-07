"""Dish Discovery: BM25F-style recipe search and metadata filters.

Reads RAW_recipes.csv from the Food.com Recipes and Interactions dataset.
No third-party dependency is required for searching; building the index uses
standard-library csv, collections, and pickle.
"""
from __future__ import annotations

import ast
import csv
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN = re.compile(r"[a-z0-9]+")
FIELDS = ("name", "description", "ingredients", "tags")
FIELD_WEIGHTS = {"name": 3.0, "description": 1.0, "ingredients": 2.0, "tags": 1.5}
DIET_TAGS = {
    "vegetarian": {"vegetarian"},
    "vegan": {"vegan"},
    "gluten-free": {"gluten-free"},
    "dairy-free": {"dairy-free"},
}


def tokenize(value: str) -> list[str]:
    return TOKEN.findall(value.lower())


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value]
    if not value:
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().lower() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(value).strip().lower()]


class RecipeSearch:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.recipes: dict[int, dict[str, Any]] = {}
        # token -> list of (recipe_id, field_name, term_frequency)
        self.postings: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        self.field_lengths: dict[int, dict[str, int]] = {}
        self.avg_lengths: dict[str, float] = {}
        self.doc_freq: dict[str, int] = {}

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "RecipeSearch":
        engine = cls()
        length_totals = Counter()
        with open(csv_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rid = int(row["id"])
                    minutes = int(float(row["minutes"]))
                except (KeyError, ValueError, TypeError):
                    continue
                # Remove corrupt/extreme duration rows if not already cleaned.
                if minutes < 0 or minutes > 10080:
                    continue
                ingredients = parse_list(row.get("ingredients"))
                tags = parse_list(row.get("tags"))
                recipe = {
                    "id": rid,
                    "name": row.get("name", ""),
                    "description": row.get("description", "") or "",
                    "ingredients": ingredients,
                    "tags": tags,
                    "minutes": minutes,
                    "steps": parse_list(row.get("steps")),
                }
                # No index-only text values here: list fields join into text.
                field_texts = {
                    "name": recipe["name"],
                    "description": recipe["description"],
                    "ingredients": " ".join(ingredients),
                    "tags": " ".join(tags),
                }
                engine.recipes[rid] = recipe
                engine.field_lengths[rid] = {}
                for field, text in field_texts.items():
                    counts = Counter(tokenize(text))
                    length = sum(counts.values())
                    engine.field_lengths[rid][field] = length
                    length_totals[field] += length
                    for term, tf in counts.items():
                        engine.postings[term].append((rid, field, tf))
        n = len(engine.recipes)
        engine.avg_lengths = {f: max(length_totals[f] / n, 1.0) if n else 1.0 for f in FIELDS}
        # Each recipe contributes only once to term DF, even across fields.
        engine.doc_freq = {
            term: len({rid for rid, _, _ in entries})
            for term, entries in engine.postings.items()
        }
        return engine

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "RecipeSearch":
        # Only load pickle files you generated and trust.
        with open(path, "rb") as handle:
            engine = pickle.load(handle)
        if not isinstance(engine, cls):
            raise TypeError("Cache is not a RecipeSearch")
        return engine

    @staticmethod
    def _matches(recipe: dict[str, Any], *, max_minutes: int | None,
                 cuisine: str | None, diet: str | None,
                 include_ingredients: list[str] | None,
                 exclude_ingredients: list[str] | None) -> bool:
        if max_minutes is not None and recipe["minutes"] > max_minutes:
            return False
        tags = set(recipe["tags"])
        # Cuisine/diet filters use the Food.com tags (not inferred nutrition).
        if cuisine and cuisine.strip().lower() not in tags:
            return False
        if diet:
            normalized = diet.strip().lower().replace("_", "-")
            if normalized not in DIET_TAGS:
                raise ValueError(f"Unknown diet {diet!r}; choose {list(DIET_TAGS)}")
            if not (tags & DIET_TAGS[normalized]):
                return False
        ingredient_text = " ".join(recipe["ingredients"])
        if include_ingredients and not all(
            phrase.lower().strip() in ingredient_text for phrase in include_ingredients
        ):
            return False
        if exclude_ingredients and any(
            phrase.lower().strip() in ingredient_text for phrase in exclude_ingredients
        ):
            return False
        return True

    def search(self, query: str, *, top_k: int = 10,
               max_minutes: int | None = None, cuisine: str | None = None,
               diet: str | None = None,
               include_ingredients: list[str] | None = None,
               exclude_ingredients: list[str] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if max_minutes is not None and max_minutes < 0:
            raise ValueError("max_minutes cannot be negative")
        if diet and diet.strip().lower().replace("_", "-") not in DIET_TAGS:
            raise ValueError(f"Unknown diet {diet!r}; choose {list(DIET_TAGS)}")
        terms = set(tokenize(query))
        n = len(self.recipes)
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            entries = self.postings.get(term, [])
            if not entries:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            # BM25F: normalize each field TF by field length, then weight and sum.
            weighted_tfs: dict[int, float] = defaultdict(float)
            for rid, field, tf in entries:
                length = self.field_lengths[rid][field]
                denom = 1 - self.b + self.b * length / self.avg_lengths[field]
                weighted_tfs[rid] += FIELD_WEIGHTS[field] * tf / denom
            for rid, weighted_tf in weighted_tfs.items():
                scores[rid] += idf * (self.k1 + 1) * weighted_tf / (self.k1 + weighted_tf)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results = []
        for rid, score in ranked:
            recipe = self.recipes[rid]
            if self._matches(recipe, max_minutes=max_minutes, cuisine=cuisine,
                             diet=diet, include_ingredients=include_ingredients,
                             exclude_ingredients=exclude_ingredients):
                results.append({**recipe, "score": round(score, 4)})
                if len(results) == top_k:
                    break
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/RAW_recipes.csv")
    parser.add_argument("--cache", default="data/bm25_index.pkl")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-minutes", type=int)
    parser.add_argument("--cuisine")
    parser.add_argument("--diet", choices=sorted(DIET_TAGS))
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    cache = Path(args.cache)
    if cache.exists():
        searcher = RecipeSearch.load(cache)
    else:
        searcher = RecipeSearch.from_csv(args.csv)
        cache.parent.mkdir(parents=True, exist_ok=True)
        searcher.save(cache)
    for result in searcher.search(
        args.query, max_minutes=args.max_minutes, cuisine=args.cuisine,
        diet=args.diet, include_ingredients=args.include,
        exclude_ingredients=args.exclude
    ):
        print(f"{result['score']:7.3f}  {result['minutes']:4} min  "
              f"{result['id']}  {result['name']}")
