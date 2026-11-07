from src.indexer import RecipeIndexer
from src.search_engine import RecipeSearch
from src.evaluator import IREvaluator

def main():
    print("Dish Discovery Core Algorithm Checkpoint")

    print("\n1. Running RecipeIndexer...")
    data_path = "data/sample_recipes.csv"
    indexer = RecipeIndexer(data_path)
    indexer.load_and_preprocess()
    print(f"   Successfully preprocessed {len(indexer.df)} recipes.")

    print("\n2. Initializing RecipeSearch Engine...")
    search_engine = RecipeSearch(indexer)
    print("   Inverted index and field length statistics built.")

    sample_query = "breakfast pizza"
    max_time = 30
    print(f"\n3. Executing Sample Search: '{sample_query}' (max_minutes={max_time})")
    results = search_engine.search(sample_query, top_k=5, max_minutes=max_time)

    print(f"   Top {len(results)} Results:")
    for rank, item in enumerate(results, start=1):
        print(f"     {rank}. [{item['score']:.4f}] ID {item['id']}: {item['name']} ({item['minutes']} mins)")

    print("\n4. Running IREvaluator...")
    benchmark_file = "data/test_queries.json"
    metrics = IREvaluator.evaluate_search_engine(
        search_engine, benchmark_file=benchmark_file, k=5
    )

    print("\n--- System Evaluation Metrics ---")
    for metric_name, score in metrics.items():
        print(f"   {metric_name}: {score}")

if __name__ == "__main__":
    main()