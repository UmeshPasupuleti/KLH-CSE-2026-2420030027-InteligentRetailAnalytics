"""
Real Dataset Ingestion, Harmonization, and Cleaning Pipeline for ConsumerIQ.
Strictly functional programming paradigm (functions only, no classes).
"""

import os
import pandas as pd
import numpy as np
from loguru import logger

TARGET_COLUMNS = [
    "dataset_source",
    "order_id",
    "customer_id",
    "order_date",
    "product_id",
    "product_category",
    "quantity",
    "unit_price",
    "total_sales_value",
]


def get_instacart_price_map() -> dict:
    """
    Returns an estimated average unit price map by department category for Instacart.
    """
    return {
        "produce": 4.50,
        "dairy eggs": 3.80,
        "snacks": 3.20,
        "beverages": 3.50,
        "frozen": 5.50,
        "pantry": 4.00,
        "bakery": 4.20,
        "canned goods": 2.80,
        "deli": 7.50,
        "dry goods pasta": 2.50,
        "household": 6.00,
        "breakfast": 4.00,
        "meat seafood": 12.00,
        "personal care": 7.00,
        "babies": 10.00,
        "international": 4.50,
        "alcohol": 14.00,
        "pets": 9.00,
        "bulk": 5.00,
        "other": 5.00,
        "missing": 5.00,
    }


def process_uci_data(filepath: str = "data/raw/uci/online_retail_II.xlsx") -> pd.DataFrame:
    """
    Extracts and harmonizes the UCI Online Retail II dataset across both years.
    Filters out cancellations and maps to target schema.
    """
    logger.info(f"Loading UCI Online Retail II from {filepath}...")
    sheet_names = ["Year 2009-2010", "Year 2010-2011"]
    frames = []

    for sheet in sheet_names:
        logger.info(f"Reading UCI sheet: {sheet}")
        df_sheet = pd.read_excel(filepath, sheet_name=sheet)
        frames.append(df_sheet)

    uci_raw = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(uci_raw):,} raw UCI records.")

    # Filter out cancelled orders (where Quantity <= 0 or Invoice starts with 'C')
    inv_series = uci_raw["Invoice"].astype(str).str.strip()
    valid_mask = (uci_raw["Quantity"] > 0) & (~inv_series.str.startswith("C")) & (uci_raw["Price"] > 0)
    uci_filtered = uci_raw[valid_mask].copy()

    # Drop records missing critical fields early
    uci_filtered = uci_filtered.dropna(subset=["Customer ID", "Description"])
    
    # Standardize types
    uci_filtered["customer_id_clean"] = uci_filtered["Customer ID"].astype(int).astype(str)
    uci_filtered["order_id_clean"] = uci_filtered["Invoice"].astype(str).str.strip()
    uci_filtered["product_id_clean"] = uci_filtered["StockCode"].astype(str).str.strip()
    uci_filtered["category_clean"] = uci_filtered["Description"].astype(str).str.strip().str.title()
    uci_filtered["quantity_clean"] = uci_filtered["Quantity"].astype(int)
    uci_filtered["price_clean"] = uci_filtered["Price"].astype(float)
    uci_filtered["total_sales"] = (uci_filtered["quantity_clean"] * uci_filtered["price_clean"]).astype(float)
    uci_filtered["order_date_clean"] = pd.to_datetime(uci_filtered["InvoiceDate"])

    uci_harmonized = pd.DataFrame({
        "dataset_source": "UCI_Online_Retail",
        "order_id": uci_filtered["order_id_clean"],
        "customer_id": uci_filtered["customer_id_clean"],
        "order_date": uci_filtered["order_date_clean"],
        "product_id": "UCI_" + uci_filtered["product_id_clean"],
        "product_category": uci_filtered["category_clean"],
        "quantity": uci_filtered["quantity_clean"],
        "unit_price": uci_filtered["price_clean"],
        "total_sales_value": uci_filtered["total_sales"],
    })

    logger.info(f"Harmonized UCI records: {len(uci_harmonized):,}")
    return uci_harmonized[TARGET_COLUMNS]


def process_instacart_data(
    base_dir: str = "data/raw/instacart",
    sample_order_count: int = 100000,
    max_items: int = 800000,
) -> pd.DataFrame:
    """
    Extracts and harmonizes the Instacart dataset.
    Merges orders, order_products__prior, products, and departments.
    Imputes department-based pricing and constructs valid datetime representations.
    """
    logger.info(f"Loading Instacart data from {base_dir}...")
    depts_file = os.path.join(base_dir, "departments.csv")
    prods_file = os.path.join(base_dir, "products.csv")
    orders_file = os.path.join(base_dir, "orders.csv")
    prior_file = os.path.join(base_dir, "order_products__prior.csv")

    departments = pd.read_csv(depts_file)
    products = pd.read_csv(prods_file, usecols=["product_id", "product_name", "department_id"])
    prod_dept = pd.merge(products, departments, on="department_id", how="left")

    logger.info("Reading Instacart orders...")
    orders = pd.read_csv(orders_file, nrows=sample_order_count if sample_order_count else None)
    
    # Calculate realistic datetime timestamps using cumulative days and hour of day
    orders["days_clean"] = orders.groupby("user_id")["days_since_prior_order"].transform(
        lambda s: s.fillna(0).cumsum()
    )
    base_date = pd.Timestamp("2017-01-01 00:00:00")
    orders["order_date_computed"] = (
        base_date
        + pd.to_timedelta(orders["days_clean"], unit="D")
        + pd.to_timedelta(orders["order_hour_of_day"], unit="h")
    )

    selected_order_ids = set(orders["order_id"])
    logger.info(f"Selected {len(selected_order_ids):,} Instacart orders. Filtering order products...")

    prior_chunks = []
    total_loaded = 0
    chunk_size = 1000000

    for chunk in pd.read_csv(
        prior_file,
        chunksize=chunk_size,
        usecols=["order_id", "product_id"],
        dtype={"order_id": "int32", "product_id": "int32"},
    ):
        matched_chunk = chunk[chunk["order_id"].isin(selected_order_ids)]
        if not matched_chunk.empty:
            prior_chunks.append(matched_chunk)
            total_loaded += len(matched_chunk)
        if max_items and total_loaded >= max_items:
            break

    prior_df = pd.concat(prior_chunks, ignore_index=True)
    if max_items and len(prior_df) > max_items:
        prior_df = prior_df.iloc[:max_items]

    logger.info(f"Merging Instacart datasets for {len(prior_df):,} product rows...")
    merged = pd.merge(
        prior_df,
        orders[["order_id", "user_id", "order_date_computed"]],
        on="order_id",
        how="inner",
    )
    merged = pd.merge(
        merged,
        prod_dept[["product_id", "department"]],
        on="product_id",
        how="left",
    )

    price_map = get_instacart_price_map()
    merged["unit_price"] = merged["department"].str.lower().map(price_map).fillna(5.00)
    merged["quantity"] = 1
    merged["total_sales_value"] = merged["unit_price"] * merged["quantity"]

    insta_harmonized = pd.DataFrame({
        "dataset_source": "Instacart",
        "order_id": merged["order_id"].astype(str),
        "customer_id": merged["user_id"].astype(str),
        "order_date": merged["order_date_computed"],
        "product_id": "INSTA_" + merged["product_id"].astype(str),
        "product_category": merged["department"].fillna("Missing").astype(str).str.title(),
        "quantity": merged["quantity"].astype(int),
        "unit_price": merged["unit_price"].astype(float),
        "total_sales_value": merged["total_sales_value"].astype(float),
    })

    logger.info(f"Harmonized Instacart records: {len(insta_harmonized):,}")
    return insta_harmonized[TARGET_COLUMNS]


def process_olist_data(base_dir: str = "data/raw/olist") -> pd.DataFrame:
    """
    Extracts and harmonizes the Olist Brazilian E-Commerce dataset.
    Merges orders, items, products, and customers.
    """
    logger.info(f"Loading Olist data from {base_dir}...")
    orders_file = os.path.join(base_dir, "olist_orders_dataset.csv")
    items_file = os.path.join(base_dir, "olist_order_items_dataset.csv")
    prods_file = os.path.join(base_dir, "olist_products_dataset.csv")
    custs_file = os.path.join(base_dir, "olist_customers_dataset.csv")

    orders = pd.read_csv(orders_file)
    items = pd.read_csv(items_file)
    products = pd.read_csv(prods_file)
    customers = pd.read_csv(custs_file)

    # Filter non-canceled orders
    valid_orders = orders[orders["order_status"] != "canceled"][
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ]

    # Merge items with orders, customers, and product categories
    m1 = pd.merge(items[["order_id", "order_item_id", "product_id", "price"]], valid_orders, on="order_id", how="inner")
    m2 = pd.merge(m1, customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")
    m3 = pd.merge(m2, products[["product_id", "product_category_name"]], on="product_id", how="left")

    # Format category names nicely (e.g. 'cama_mesa_banho' -> 'Cama Mesa Banho')
    category_formatted = (
        m3["product_category_name"]
        .fillna("Other")
        .astype(str)
        .str.replace("_", " ")
        .str.title()
    )

    olist_harmonized = pd.DataFrame({
        "dataset_source": "Olist",
        "order_id": m3["order_id"].astype(str),
        "customer_id": m3["customer_unique_id"].astype(str),
        "order_date": pd.to_datetime(m3["order_purchase_timestamp"]),
        "product_id": "OLIST_" + m3["product_id"].astype(str),
        "product_category": category_formatted,
        "quantity": 1,
        "unit_price": m3["price"].astype(float),
        "total_sales_value": m3["price"].astype(float),
    })

    logger.info(f"Harmonized Olist records: {len(olist_harmonized):,}")
    return olist_harmonized[TARGET_COLUMNS]


def clean_unified_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the unified dataset:
    - Drops rows missing critical fields (customer_id, product_category).
    - Standardizes order_date into standard datetime format (YYYY-MM-DD HH:MM:SS).
    - Removes outliers and negative price/quantity entries.
    """
    logger.info(f"Starting unified data cleaning on {len(df):,} records...")
    initial_count = len(df)

    # Remove nulls in critical columns
    critical_cols = ["customer_id", "product_category", "order_id", "product_id", "order_date"]
    df_clean = df.dropna(subset=critical_cols).copy()

    # Filter out empty or placeholder strings
    invalid_placeholders = {"nan", "none", "null", "missing", "unknown", ""}
    for col in ["customer_id", "product_category", "order_id", "product_id"]:
        df_clean = df_clean[~df_clean[col].astype(str).str.strip().str.lower().isin(invalid_placeholders)]

    # Standardize order_date format
    df_clean["order_date"] = pd.to_datetime(df_clean["order_date"], errors="coerce")
    df_clean = df_clean.dropna(subset=["order_date"])
    df_clean["order_date"] = df_clean["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove outliers and negative / zero price and quantity
    df_clean["quantity"] = pd.to_numeric(df_clean["quantity"], errors="coerce")
    df_clean["unit_price"] = pd.to_numeric(df_clean["unit_price"], errors="coerce")
    df_clean = df_clean.dropna(subset=["quantity", "unit_price"])

    valid_mask = (
        (df_clean["quantity"] > 0)
        & (df_clean["quantity"] <= 10000)
        & (df_clean["unit_price"] > 0.0)
        & (df_clean["unit_price"] <= 100000.0)
    )
    df_clean = df_clean[valid_mask].copy()

    # Recalculate total_sales_value for consistency and precision
    df_clean["total_sales_value"] = (df_clean["quantity"] * df_clean["unit_price"]).round(2)
    df_clean["unit_price"] = df_clean["unit_price"].round(2)

    # Ensure schema column order and string types
    df_clean = df_clean[TARGET_COLUMNS].reset_index(drop=True)
    logger.info(
        f"Cleaning completed: {len(df_clean):,} records retained ({initial_count - len(df_clean):,} removed)."
    )
    return df_clean


from src.graph_etl_pipeline import (
    build_hetero_data,
    save_hetero_graph,
    run_graph_etl_pipeline,
)


def run_pipeline(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/unified_consumeriq_real.csv",
    generate_graph: bool = True,
    graph_output_path: str = "data/processed/consumeriq_hetero_graph.pt",
) -> pd.DataFrame:
    """
    Executes end-to-end extraction, harmonization, cleaning, and persistence.
    DEPRECATION NOTICE: Flat RFM metric extraction is deprecated in favor of graph-structured
    heterogeneous representations for Graph Neural Networks (GNNs).
    When generate_graph is True, parses transaction logs into a PyTorch Geometric HeteroData object.
    """
    logger.info("=== Starting ConsumerIQ Real Data ETL Pipeline ===")
    
    uci_path = os.path.join(raw_dir, "uci", "online_retail_II.xlsx")
    insta_dir = os.path.join(raw_dir, "instacart")
    olist_dir = os.path.join(raw_dir, "olist")

    df_uci = process_uci_data(uci_path)
    df_insta = process_instacart_data(insta_dir)
    df_olist = process_olist_data(olist_dir)

    logger.info("Concatenating all harmonized datasets...")
    unified_raw = pd.concat([df_uci, df_insta, df_olist], ignore_index=True)
    logger.info(f"Combined total before cleaning: {len(unified_raw):,} records.")

    unified_clean = clean_unified_data(unified_raw)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving cleaned dataset to {output_path}...")
    unified_clean.to_csv(output_path, index=False)
    logger.info(f"Successfully exported {len(unified_clean):,} rows to {output_path}")

    # Log summary statistics
    summary = unified_clean.groupby("dataset_source").agg(
        total_rows=("order_id", "count"),
        total_orders=("order_id", "nunique"),
        total_customers=("customer_id", "nunique"),
        total_revenue=("total_sales_value", "sum"),
    )
    logger.info(f"Pipeline Summary:\n{summary}")

    # Graph Neural Network ETL Execution (HeteroData)
    if generate_graph:
        logger.info("Overhauling pipeline: constructing PyTorch Geometric HeteroData graph...")
        graph_data, _ = build_hetero_data(unified_clean)
        save_hetero_graph(graph_data, filepath=graph_output_path)
        logger.info(f"HeteroData graph generated and persisted at {graph_output_path}")

    return unified_clean


if __name__ == "__main__":
    run_pipeline()
