import os
import pickle
import urllib.request
import streamlit as st
from src.indexer import RecipeIndexer
from src.search_engine import RecipeSearch

st.set_page_config(
    page_title="Dish Discovery",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: 700; color: #1E293B; margin-bottom: 0.25rem; }
.sub-header { font-size: 1rem; color: #64748B; margin-bottom: 2rem; }
.recipe-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.score-badge {
    position: relative;
    background-color: #EFF6FF;
    color: #2563EB;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.85rem;
    cursor: pointer;
}
.score-badge .tooltip-text {
    visibility: hidden;
    background-color: #1E293B;
    color: #FFFFFF;
    text-align: center;
    border-radius: 6px;
    padding: 4px 8px;
    position: absolute;
    z-index: 10;
    bottom: 125%;
    right: 0;
    white-space: nowrap;
    font-size: 0.75rem;
    font-weight: 400;
    opacity: 0;
    transition: opacity 0.15s;
}
.score-badge:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
.tag-chip {
    display: inline-block;
    background-color: #F1F5F9;
    color: #475569;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "data/RAW_recipes.csv"
CACHE_FILE = "data/search_engine.pkl"
DATASET_URL = "https://github.com/ruthjasa/DishDiscovery/releases/download/v1.0-data/RAW_recipes.csv"


@st.cache_resource
def init_pipeline():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            indexer, search_engine = pickle.load(f)
        return indexer, search_engine

    with st.spinner("Indexing 230k+ recipes for the first time (subsequent launches will take <1s)..."):
        indexer = RecipeIndexer(DATA_FILE)
        indexer.load_and_preprocess()
        search_engine = RecipeSearch(indexer)

        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((indexer, search_engine), f, protocol=pickle.HIGHEST_PROTOCOL)

    return indexer, search_engine


indexer, search_engine = init_pipeline()

st.markdown('<div class="main-header">🍽️ Dish Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Field-Weighted BM25F Recipe Search & Pantry Matching Engine</div>', unsafe_allow_html=True)

st.sidebar.header("🔍 Search & Pantry Filters")

max_time = st.sidebar.slider("Max Preparation Time (mins)", min_value=5, max_value=120, value=45, step=5)

diet_options = ["vegetarian", "vegan", "gluten-free", "dairy-free"]
selected_diets = st.sidebar.multiselect("Dietary Preferences", diet_options)

common_cuisines = [
    "Any", "italian", "mexican", "asian", "american",
    "indian", "mediterranean", "french", "thai", "chinese"
]
selected_cuisine = st.sidebar.selectbox("Cuisine Type", common_cuisines)
cuisine_param = None if selected_cuisine == "Any" else selected_cuisine

include_ing = st.sidebar.text_input("Pantry / Include Ingredients (comma-separated)", placeholder="e.g. chicken, garlic, rice")
exclude_ing = st.sidebar.text_input("Exclude Ingredients (comma-separated)", placeholder="e.g. peanuts, milk")

inc_list = [i.strip() for i in include_ing.split(",") if i.strip()] or None
exc_list = [e.strip() for e in exclude_ing.split(",") if e.strip()] or None

query = st.text_input("Enter keywords, ingredients, or dish names:", placeholder="e.g. 'breakfast pizza' or 'quick garlic pasta'")

if query:
    try:
        primary_diet = selected_diets[0] if selected_diets else None

        raw_results = search_engine.search(
            query=query,
            top_k=30 if len(selected_diets) > 1 else 10,
            max_minutes=max_time,
            cuisine=cuisine_param,
            diet=primary_diet,
            include_ingredients=inc_list,
            exclude_ingredients=exc_list
        )

        if len(selected_diets) > 1:
            results = []
            for r in raw_results:
                recipe_tags = [t.lower() for t in r.get("tags", [])]
                if all(d.lower() in recipe_tags for d in selected_diets):
                    results.append(r)
                if len(results) == 10:
                    break
        else:
            results = raw_results

        st.markdown(f"### Results ({len(results)} matches found)")

        if not results:
            st.warning("No recipes matched your exact query and filter criteria. Try relaxing your time, dietary, or ingredient filters!")

        max_score = max((r["score"] for r in results), default=1.0) or 1.0

        for r in results:
            match_pct = int(round((r["score"] / max_score) * 100))

            row_match = indexer.df[indexer.df["id"] == r["id"]]
            full_row = row_match.iloc[0] if not row_match.empty else {}

            raw_ings = r.get("ingredients") if r.get("ingredients") else full_row.get("ingredients", [])
            ingredients_list = (
                RecipeIndexer.parse_list(raw_ings)
                if isinstance(raw_ings, str)
                else (list(raw_ings) if raw_ings is not None else [])
            )

            raw_steps = r.get("steps") if r.get("steps") else full_row.get("steps", [])
            steps_list = (
                RecipeIndexer.parse_list(raw_steps)
                if isinstance(raw_steps, str)
                else (list(raw_steps) if raw_steps is not None else [])
            )

            pantry_badge_html = ""
            if inc_list:
                matched_ings = [
                    ing for ing in inc_list
                    if any(ing.lower() in r_ing.lower() for r_ing in ingredients_list)
                ]
                extra_to_buy = max(0, len(ingredients_list) - len(matched_ings))
                pantry_badge_html = (
                    f'<span class="tag-chip" style="background-color:#DCFCE7; color:#166534;">'
                    f'🧺 Pantry Match: {len(matched_ings)}/{len(inc_list)} requested '
                    f'({extra_to_buy} extra items needed)</span>'
                )

            diet_badges_html = "".join(
                f'<span class="tag-chip" style="background-color:#FEF3C7; color:#92400E;">✓ {d}</span>'
                for d in selected_diets
            )

            # Only render the badge row if there are active pantry or diet badges
            badges_html = ""
            if pantry_badge_html or diet_badges_html:
                badges_html = f'<div style="margin-top: 0.35rem;">{pantry_badge_html}{diet_badges_html}</div>'

            # Render Recipe Card (flush-left HTML avoids Markdown code-block formatting)
            card_html = (
                f'<div class="recipe-card">'
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<h3 style="margin:0; font-size: 1.2rem; color: #0F172A;">{r["name"].title()}</h3>'
                f'<span class="score-badge">{match_pct}% Match'
                f'<span class="tooltip-text">Raw BM25F Score: {r["score"]:.2f}</span>'
                f'</span>'
                f'</div>'
                f'<p style="color: #64748B; font-size: 0.9rem; margin: 0.4rem 0 0.2rem 0;">'
                f'⏱️ {r["minutes"]} minutes &nbsp;|&nbsp; 🥘 {len(ingredients_list)} total ingredients'
                f'</p>'
                f'{badges_html}'
                f'</div>'
            )

            with st.container():
                st.markdown(card_html, unsafe_allow_html=True)

                # Side-by-side Expanders for Ingredients & Cooking Steps
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander(f"🥕 Ingredients ({len(ingredients_list)})"):
                        if ingredients_list:
                            for item in ingredients_list:
                                st.markdown(f"- {item}")
                        else:
                            st.write("No ingredients listed.")

                with col2:
                    with st.expander(f"📋 Cooking Instructions ({len(steps_list)} steps)"):
                        if steps_list:
                            for idx, step in enumerate(steps_list, start=1):
                                st.markdown(f"**{idx}.** {step.capitalize()}")
                        else:
                            st.write("No instructions listed.")

    except ValueError as e:
        st.error(f"Invalid Search Parameter: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while executing search: {e}")
else:
    st.info("Type a query above to execute BM25F ranked retrieval and pantry matching.")