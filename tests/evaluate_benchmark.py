import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.indexer import RecipeIndexer
from src.search_engine import RecipeSearch

DATA_FILE = "data/RAW_recipes.csv"
CACHE_FILE = "data/search_engine.pkl"
TOP_K = 10

def load_system():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached index from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"Building index from {DATA_FILE}...")
    indexer = RecipeIndexer(DATA_FILE)
    indexer.load_and_preprocess()
    engine = RecipeSearch(indexer)
    return indexer, engine

class TfidfBaselineSearch:
    def __init__(self, indexer: RecipeIndexer):
        self.df = indexer.df.copy()
        self.recipe_ids = self.df["id"].tolist()

        # Safely combine whichever text columns exist in indexer.df
        text_cols = [c for c in ["name", "ingredients", "tags", "description", "steps"] if c in self.df.columns]
        corpus = (
            self.df[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = self.df.iloc[idx]
            results.append({
                "id": row["id"],
                "name": row["name"],
                "minutes": row["minutes"],
                "tags": row["tags"],
                "ingredients": row["ingredients"],
                "score": float(scores[idx]),
            })
        return results

BENCHMARK_QUERIES = [
    {"query": "breakfast pizza", "keywords": ["breakfast", "pizza"], "max_mins": None, "diet": None},
    {"query": "quick garlic pasta", "keywords": ["garlic", "pasta"], "max_mins": 30, "diet": None},
    {"query": "vegan black bean tacos", "keywords": ["black bean", "taco"], "max_mins": None, "diet": "vegan"},
    {"query": "30 minute chicken stir fry", "keywords": ["chicken", "stir fry"], "max_mins": 30, "diet": None},
    {"query": "gluten free banana bread", "keywords": ["banana", "bread"], "max_mins": None, "diet": "gluten-free"},
    {"query": "dairy free creamy tomato soup", "keywords": ["tomato", "soup"], "max_mins": None, "diet": "dairy-free"},
    {"query": "easy microwave mug cake", "keywords": ["mug", "cake"], "max_mins": 15, "diet": None},
    {"query": "vegetarian spinach lasagna", "keywords": ["spinach", "lasagna"], "max_mins": None, "diet": "vegetarian"},
    {"query": "spicy honey garlic salmon", "keywords": ["honey", "garlic", "salmon"], "max_mins": None, "diet": None},
    {"query": "5 ingredient peanut butter cookies", "keywords": ["peanut butter", "cookie"], "max_mins": None, "diet": None},
    {"query": "15 minute egg fried rice", "keywords": ["egg", "fried rice"], "max_mins": 15, "diet": None},
    {"query": "slow cooker beef chili", "keywords": ["beef", "chili"], "max_mins": None, "diet": None},
    {"query": "mediterranean chickpea salad", "keywords": ["chickpea", "salad"], "max_mins": None, "diet": None},
    {"query": "dairy free chocolate chip cookies", "keywords": ["chocolate chip", "cookie"], "max_mins": None, "diet": "dairy-free"},
    {"query": "crispy baked tofu bowl", "keywords": ["tofu"], "max_mins": None, "diet": "vegan"},
    {"query": "cheesy broccoli potato soup", "keywords": ["broccoli", "potato", "soup"], "max_mins": None, "diet": None},
    {"query": "quick avocado toast with egg", "keywords": ["avocado", "toast", "egg"], "max_mins": 15, "diet": None},
    {"query": "gluten free pancakes", "keywords": ["pancake"], "max_mins": None, "diet": "gluten-free"},
    {"query": "one pot lemon herb chicken", "keywords": ["lemon", "chicken"], "max_mins": 45, "diet": None},
    {"query": "vegan lentil curry", "keywords": ["lentil", "curry"], "max_mins": None, "diet": "vegan"},
]


def judge_relevance(recipe: dict, spec: dict) -> int:
    """
    Graded relevance judgment (0, 1, 2):
      - 2 (Highly Relevant): Matches title keywords AND satisfies time/diet constraints.
      - 1 (Partially Relevant): Matches keywords in body/ingredients OR misses soft constraint.
      - 0 (Not Relevant): Fails core topic match or violates hard dietary restriction.
    """
    name = str(recipe.get("name", "")).lower()
    ings = " ".join(recipe.get("ingredients", [])).lower() if isinstance(recipe.get("ingredients"), list) else str(recipe.get("ingredients", "")).lower()
    tags = [t.lower() for t in recipe.get("tags", [])] if isinstance(recipe.get("tags"), list) else []
    minutes = recipe.get("minutes", 999)

    if spec["diet"] and spec["diet"].lower() not in tags:
        return 0

    time_ok = True if spec["max_mins"] is None else (minutes <= spec["max_mins"])

    title_hits = sum(1 for kw in spec["keywords"] if kw in name)
    body_hits = sum(1 for kw in spec["keywords"] if kw in name or kw in ings)

    if title_hits == len(spec["keywords"]) and time_ok:
        return 2
    elif body_hits >= max(1, len(spec["keywords"]) - 1) and time_ok:
        return 1
    return 0

def precision_at_k(relevances: list[int], k: int = 10) -> float:
    rel_k = relevances[:k]
    if not rel_k:
        return 0.0
    return sum(1 for r in rel_k if r > 0) / k


def average_precision(relevances: list[int], k: int = 10) -> float:
    rel_k = relevances[:k]
    hits = 0
    sum_precisions = 0.0
    for idx, r in enumerate(rel_k, start=1):
        if r > 0:
            hits += 1
            sum_precisions += hits / idx
    return sum_precisions / hits if hits > 0 else 0.0


def ndcg_at_k(relevances: list[int], ideal_relevances: list[int], k: int = 10) -> float:
    def dcg(scores):
        return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores[:k]))

    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(sorted(ideal_relevances, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def run_benchmark():
    indexer, bm25f_engine = load_system()
    print("Fitting TF-IDF Cosine Similarity baseline...")
    tfidf_engine = TfidfBaselineSearch(indexer)

    id_to_row = indexer.df.set_index("id").to_dict(orient="index")

    tfidf_metrics = {"P@10": [], "MAP": [], "NDCG@10": []}
    bm25f_metrics = {"P@10": [], "MAP": [], "NDCG@10": []}
    per_query_rows = []

    for item in BENCHMARK_QUERIES:
        q = item["query"]

        tfidf_res = tfidf_engine.search(q, top_k=TOP_K)
        bm25f_res = bm25f_engine.search(
            query=q,
            top_k=TOP_K,
            max_minutes=item["max_mins"],
            diet=item["diet"]
        )

        def enrich(res_list):
            enriched = []
            for r in res_list:
                meta = id_to_row.get(r["id"], {})
                merged = {**meta, **r}
                enriched.append(merged)
            return enriched

        tfidf_enriched = enrich(tfidf_res)
        bm25f_enriched = enrich(bm25f_res)

        tfidf_rels = [judge_relevance(r, item) for r in tfidf_enriched]
        bm25f_rels = [judge_relevance(r, item) for r in bm25f_enriched]

        pooled_rels = tfidf_rels + bm25f_rels

        t_p10 = precision_at_k(tfidf_rels, TOP_K)
        t_map = average_precision(tfidf_rels, TOP_K)
        t_ndcg = ndcg_at_k(tfidf_rels, pooled_rels, TOP_K)

        b_p10 = precision_at_k(bm25f_rels, TOP_K)
        b_map = average_precision(bm25f_rels, TOP_K)
        b_ndcg = ndcg_at_k(bm25f_rels, pooled_rels, TOP_K)

        tfidf_metrics["P@10"].append(t_p10)
        tfidf_metrics["MAP"].append(t_map)
        tfidf_metrics["NDCG@10"].append(t_ndcg)

        bm25f_metrics["P@10"].append(b_p10)
        bm25f_metrics["MAP"].append(b_map)
        bm25f_metrics["NDCG@10"].append(b_ndcg)

        per_query_rows.append({
            "Query": q,
            "TFIDF_P@10": round(t_p10, 3),
            "BM25F_P@10": round(b_p10, 3),
            "TFIDF_NDCG@10": round(t_ndcg, 3),
            "BM25F_NDCG@10": round(b_ndcg, 3),
        })

    summary_df = pd.DataFrame({
        "Retrieval Model": ["TF-IDF + Cosine Similarity (Baseline)", "Field-Weighted BM25F + Filters (Ours)"],
        "Precision@10": [round(np.mean(tfidf_metrics["P@10"]), 4), round(np.mean(bm25f_metrics["P@10"]), 4)],
        "MAP": [round(np.mean(tfidf_metrics["MAP"]), 4), round(np.mean(bm25f_metrics["MAP"]), 4)],
        "NDCG@10": [round(np.mean(tfidf_metrics["NDCG@10"]), 4), round(np.mean(bm25f_metrics["NDCG@10"]), 4)],
    })

    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS ACROSS 20 QUERIES")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("=" * 72)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(per_query_rows).to_csv("results/benchmark_comparison.csv", index=False)
    print("\nSaved per-query comparison table to 'results/benchmark_comparison.csv'.")

def load_system():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached index from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"Building index from {DATA_FILE}...")
    indexer = RecipeIndexer(DATA_FILE)
    indexer.load_and_preprocess()
    engine = RecipeSearch(indexer)

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((indexer, engine), f, protocol=pickle.HIGHEST_PROTOCOL)

    return indexer, engine


if __name__ == "__main__":
    run_benchmark()