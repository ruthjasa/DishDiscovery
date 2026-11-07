# Dish Discovery — Information Retrieval Pipeline

An end-to-end Information Retrieval system for recipe search using field-weighted **BM25F** ranking, metadata filtering, and quantitative benchmark evaluation.

---

## Environment Setup & Installation

Follow these steps to set up the environment and install project dependencies locally.

### Prerequisites
* **Python 3.9+** (Python 3.10–3.12 recommended)
* **Git**

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/DishDiscovery.git](https://github.com/YOUR_USERNAME/DishDiscovery.git)
cd DishDiscovery
```

### 2. Create and Activate a Virtual Environment
- Recommended, but not required
- Windows:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
- macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
- Install required libraries (pandas, numpy, nltk, rank-bm25) specified in requirements.txt
```bash
pip install -r requirements.txt
```

---

## Running the System
- main.py is the main entry point for Dish Discovery
- Integrates RecipeIndexer, RecipeSearch, and IREvaluator together
- Run the pipeline from the project root directory using:
```bash
python main.py
```

### Expected Output
- 100 recipes are successfully preprocessed
- The inverted index and field length statistics are built
- Sample search for 'breakfast pizza' with the parameter of 30 minutes max cooking time yields 2 results, IDs being 31490 & 52443
- The evaluator successfully outputs performance metrics for Mean Precision @ 5, MAP, and Mean NDCG @ 5

---

## Running Automated Tests
- We utilized unittest to ensure that the system functions properly
- To run the full test suite from the project root:
```bash
python -m unittest discover -s tests
```
- To test specific unit test files:
```bash
python -m unittest tests/test_evaluator.py
python -m unittest tests/test_search.py
python -m unittest tests/test_indexing.py
```