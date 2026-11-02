"""
Market Basket Analysis and Association Rule Mining for ConsumerIQ.
Strictly functional programming paradigm (functions only, no classes).
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from loguru import logger


def filter_basket_data(
    df: pd.DataFrame,
    top_n_categories: int = 50,
    max_orders: int = 50000,
) -> pd.DataFrame:
    """
    Optimizes memory and computational performance:
    - Filters to top N most frequent product categories.
    - Selects up to max_orders most recent transactions.
    """
    logger.info(f"Filtering dataset for Market Basket Analysis (top {top_n_categories} categories, max {max_orders:,} orders)...")

    # Filter top categories
    top_categories = df["product_category"].value_counts().head(top_n_categories).index
    df_filtered = df[df["product_category"].isin(top_categories)].copy()

    # Select most recent orders if order_date is present, else first unique orders
    if "order_date" in df_filtered.columns:
        if not np.issubdtype(df_filtered["order_date"].dtype, np.datetime64):
            df_filtered["order_date"] = pd.to_datetime(df_filtered["order_date"])
        
        recent_order_ids = (
            df_filtered.groupby("order_id")["order_date"]
            .max()
            .sort_values(ascending=False)
            .head(max_orders)
            .index
        )
    else:
        recent_order_ids = df_filtered["order_id"].drop_duplicates().head(max_orders)

    df_sample = df_filtered[df_filtered["order_id"].isin(recent_order_ids)].copy()
    logger.info(f"Retained {len(df_sample):,} items across {len(recent_order_ids):,} orders.")
    return df_sample


def create_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a binary basket matrix (rows = order_id, columns = product_category, values = True/False).
    """
    logger.info("Constructing one-hot encoded basket matrix...")
    basket = (
        df.groupby(["order_id", "product_category"], observed=True)["quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    # Convert to boolean for memory-efficient apriori processing
    basket_bool = basket > 0
    logger.info(f"Created basket matrix with shape: {basket_bool.shape}")
    return basket_bool


def format_rules_dataframe(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and formats frozensets and numerical fields in association rules DataFrame.
    """
    if rules.empty:
        return rules

    formatted = rules.copy()
    formatted["antecedents"] = formatted["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    formatted["consequents"] = formatted["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    
    # Round numerical metrics
    formatted["support"] = formatted["support"].round(4)
    formatted["confidence"] = formatted["confidence"].round(4)
    formatted["lift"] = formatted["lift"].round(3)
    if "leverage" in formatted.columns:
        formatted["leverage"] = formatted["leverage"].round(4)
    if "conviction" in formatted.columns:
        formatted["conviction"] = formatted["conviction"].round(3)

    return formatted


def generate_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.01,
    min_threshold: float = 1.0,
    top_n_categories: int = 50,
    max_orders: int = 50000,
) -> pd.DataFrame:
    """
    Performs end-to-end Market Basket Analysis:
    1. Filters data by top categories and order sampling.
    2. Builds one-hot basket matrix.
    3. Runs Apriori algorithm for frequent itemsets.
    4. Computes association rules (Support, Confidence, Lift).
    5. Returns rules sorted by Lift descending.
    """
    logger.info(f"Generating association rules with min_support={min_support}, min_lift={min_threshold}...")
    
    filtered_df = filter_basket_data(
        df,
        top_n_categories=top_n_categories,
        max_orders=max_orders,
    )

    basket_matrix = create_basket_matrix(filtered_df)

    # Run Apriori
    try:
        frequent_itemsets = apriori(
            basket_matrix,
            min_support=min_support,
            use_colnames=True,
            low_memory=True,
        )
    except Exception as exc:
        logger.warning(f"Apriori run with min_support={min_support} encountered error: {exc}. Retrying with min_support=0.005")
        frequent_itemsets = apriori(
            basket_matrix,
            min_support=0.005,
            use_colnames=True,
            low_memory=True,
        )

    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found with current min_support threshold.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    logger.info(f"Found {len(frequent_itemsets):,} frequent itemsets. Computing association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    # Sort rules by Lift descending
    rules_sorted = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    rules_clean = format_rules_dataframe(rules_sorted)

    logger.info(f"Generated {len(rules_clean):,} association rules. Max lift: {rules_clean['lift'].max() if not rules_clean.empty else 0}")
    return rules_clean
