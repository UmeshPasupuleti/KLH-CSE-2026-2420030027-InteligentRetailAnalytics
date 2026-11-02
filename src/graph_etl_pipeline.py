"""
Graph-Based ETL Pipeline for ConsumerIQ.
Overhauls flat RFM extraction to construct a PyTorch Geometric HeteroData bipartite graph.
Strictly functional programming paradigm (functions only, no classes).
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from loguru import logger


def create_node_mappings(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    """
    Extracts unique customer and product identifiers and creates 0-indexed contiguous integer mappings.
    Returns: (customer_to_idx, product_to_idx, unique_customers, unique_products)
    """
    logger.info("Generating contiguous integer index mappings for customer and product nodes...")
    
    unique_customers = df["customer_id"].astype(str).unique()
    unique_products = df["product_id"].astype(str).unique()

    customer_to_idx = {cid: idx for idx, cid in enumerate(unique_customers)}
    product_to_idx = {pid: idx for idx, pid in enumerate(unique_products)}

    logger.info(
        f"Indexed {len(unique_customers):,} unique customer nodes and "
        f"{len(unique_products):,} unique product nodes."
    )
    return customer_to_idx, product_to_idx, unique_customers, unique_products


def extract_customer_node_features(
    df: pd.DataFrame,
    customer_to_idx: dict[str, int],
) -> torch.Tensor:
    """
    Computes normalized behavioral feature vectors for customer nodes.
    Features:
      0: log1p(total_quantity) - lifetime purchasing volume
      1: log1p(total_spend) - lifetime monetary revenue contribution
      2: log1p(unique_products) - customer breadth / distinct items purchased (node degree)
      3: log1p(avg_item_spend) - average spending per item purchased
    Features are z-score standardized and sorted by contiguous customer integer IDs.
    Returns: torch.FloatTensor of shape (num_customers, 4)
    """
    logger.info("Computing customer node behavioral feature tensors...")
    
    cust_agg = (
        df.groupby("customer_id", observed=True)
        .agg(
            total_qty=("quantity", "sum"),
            total_spend=("total_sales_value", "sum"),
            unique_prods=("product_id", "nunique"),
        )
    )
    cust_agg["avg_item_spend"] = cust_agg["total_spend"] / cust_agg["total_qty"].clip(lower=1)

    # Reindex to match contiguous customer_to_idx order exactly
    ordered_cids = sorted(customer_to_idx.keys(), key=lambda c: customer_to_idx[c])
    cust_agg_ordered = cust_agg.reindex(ordered_cids).fillna(0.0)

    raw_features = cust_agg_ordered[["total_qty", "total_spend", "unique_prods", "avg_item_spend"]].values
    
    # Skewness correction via log1p
    log_features = np.log1p(np.clip(raw_features, 0.0, None))
    
    # Z-score standardization
    mean = np.mean(log_features, axis=0)
    std = np.std(log_features, axis=0) + 1e-8
    norm_features = (log_features - mean) / std

    tensor_features = torch.tensor(norm_features, dtype=torch.float32)
    logger.info(f"Customer node features tensor shape: {tensor_features.shape}")
    return tensor_features


def extract_product_node_features(
    df: pd.DataFrame,
    product_to_idx: dict[str, int],
) -> torch.Tensor:
    """
    Computes normalized product catalog feature vectors for product nodes.
    Features:
      0: log1p(median_unit_price) - normalized price tier of the item
      1: log1p(total_quantity_sold) - item sales velocity / popularity
      2: log1p(total_revenue) - cumulative gross revenue generated
      3: log1p(unique_buyers) - customer penetration / distinct buyer count (node degree)
    Features are z-score standardized and sorted by contiguous product integer IDs.
    Returns: torch.FloatTensor of shape (num_products, 4)
    """
    logger.info("Computing product node catalog feature tensors...")
    
    prod_agg = (
        df.groupby("product_id", observed=True)
        .agg(
            median_price=("unit_price", "median"),
            total_qty=("quantity", "sum"),
            total_rev=("total_sales_value", "sum"),
            unique_buyers=("customer_id", "nunique"),
        )
    )

    # Reindex to match contiguous product_to_idx order exactly
    ordered_pids = sorted(product_to_idx.keys(), key=lambda p: product_to_idx[p])
    prod_agg_ordered = prod_agg.reindex(ordered_pids).fillna(0.0)

    raw_features = prod_agg_ordered[["median_price", "total_qty", "total_rev", "unique_buyers"]].values

    # Skewness correction via log1p
    log_features = np.log1p(np.clip(raw_features, 0.0, None))

    # Z-score standardization
    mean = np.mean(log_features, axis=0)
    std = np.std(log_features, axis=0) + 1e-8
    norm_features = (log_features - mean) / std

    tensor_features = torch.tensor(norm_features, dtype=torch.float32)
    logger.info(f"Product node features tensor shape: {tensor_features.shape}")
    return tensor_features


def extract_user_item_edges(
    df: pd.DataFrame,
    customer_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parses user-item transaction logs into bipartite edges weighted by normalized frequency and monetary value.
    - edge_index: LongTensor of shape (2, num_edges) mapping customer -> product.
    - edge_attr: FloatTensor of shape (num_edges, 2) containing [norm_frequency, norm_monetary].
    - edge_weight: FloatTensor of shape (num_edges,) composite scalar interaction weight.
    """
    logger.info("Aggregating user-item purchase interactions and computing edge weights...")
    
    # Aggregate transaction logs per (customer, product) dyad
    interactions = (
        df.groupby(["customer_id", "product_id"], observed=True)
        .agg(
            frequency=("quantity", "sum"),
            monetary=("total_sales_value", "sum"),
        )
        .reset_index()
    )

    # Map to integer node indices
    src_indices = interactions["customer_id"].astype(str).map(customer_to_idx).values
    dst_indices = interactions["product_id"].astype(str).map(product_to_idx).values

    edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)

    # Extract raw edge interaction weights
    freq_raw = interactions["frequency"].clip(lower=1.0).values
    mon_raw = interactions["monetary"].clip(lower=0.0).values

    # Non-linear log1p scaling to compress extreme heavy-tailed purchase spikes
    log_freq = np.log1p(freq_raw)
    norm_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-8)

    log_mon = np.log1p(mon_raw)
    norm_mon = (log_mon - log_mon.min()) / (log_mon.max() - log_mon.min() + 1e-8)

    # Multi-dimensional edge attributes [normalized_frequency, normalized_monetary]
    edge_attr = torch.tensor(
        np.column_stack([norm_freq, norm_mon]),
        dtype=torch.float32,
    )

    # Composite scalar interaction weight (balanced 50% purchase volume + 50% monetary spend)
    edge_weight = torch.tensor(
        0.5 * norm_freq + 0.5 * norm_mon,
        dtype=torch.float32,
    )

    logger.info(
        f"Extracted {edge_index.shape[1]:,} unique bipartite edges. "
        f"Mean norm frequency: {norm_freq.mean():.4f}, Mean norm monetary: {norm_mon.mean():.4f}"
    )
    return edge_index, edge_attr, edge_weight


def build_hetero_data(df: pd.DataFrame) -> tuple[HeteroData, dict]:
    """
    Parses flat e-commerce transaction logs into a PyTorch Geometric HeteroData object.
    Constructs:
      - Distinct node types: 'customer' and 'product'
      - Bipartite purchase edges: ('customer', 'buys', 'product')
      - Reverse edges for bidirectional GNN message passing: ('product', 'rev_buys', 'customer')
      - Edge weights normalized by purchase frequency and gross monetary spend.
    Returns: (hetero_data, node_index_mappings)
    """
    logger.info("=== Transforming Transaction Logs to PyTorch Geometric HeteroData ===")
    start_time = time.time()

    # 1. Generate contiguous integer index mappings
    customer_to_idx, product_to_idx, unique_customers, unique_products = create_node_mappings(df)

    # 2. Extract node feature matrices
    customer_x = extract_customer_node_features(df, customer_to_idx)
    product_x = extract_product_node_features(df, product_to_idx)

    # 3. Extract weighted bipartite edges
    edge_index, edge_attr, edge_weight = extract_user_item_edges(df, customer_to_idx, product_to_idx)

    # 4. Instantiate PyTorch Geometric HeteroData
    data = HeteroData()

    # Customer node type
    data["customer"].num_nodes = len(unique_customers)
    data["customer"].node_id = torch.arange(len(unique_customers), dtype=torch.long)
    data["customer"].x = customer_x

    # Product node type
    data["product"].num_nodes = len(unique_products)
    data["product"].node_id = torch.arange(len(unique_products), dtype=torch.long)
    data["product"].x = product_x

    # Forward edge relation: ('customer', 'buys', 'product')
    data["customer", "buys", "product"].edge_index = edge_index
    data["customer", "buys", "product"].edge_attr = edge_attr
    data["customer", "buys", "product"].edge_weight = edge_weight

    # Reverse edge relation: ('product', 'rev_buys', 'customer')
    data["product", "rev_buys", "customer"].edge_index = edge_index.flip(0)
    data["product", "rev_buys", "customer"].edge_attr = edge_attr
    data["product", "rev_buys", "customer"].edge_weight = edge_weight

    mappings = {
        "customer_to_idx": customer_to_idx,
        "product_to_idx": product_to_idx,
        "unique_customers": unique_customers,
        "unique_products": unique_products,
    }

    elapsed = time.time() - start_time
    logger.info(f"Constructed HeteroData graph in {elapsed:.2f} seconds.")
    logger.info(f"Graph Schema Summary:\n{data}")
    return data, mappings


def save_hetero_graph(
    data: HeteroData,
    filepath: str = "data/processed/consumeriq_hetero_graph.pt",
    metadata: dict = None,
) -> None:
    """
    Serializes and persists the HeteroData graph object and index metadata to disk.
    """
    logger.info(f"Saving PyTorch Geometric HeteroData graph to {filepath}...")
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    save_payload = {
        "graph": data,
        "metadata": metadata or {},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(save_payload, filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"Successfully saved HeteroData graph ({file_size_mb:.2f} MB) to {filepath}")


def load_hetero_graph(
    filepath: str = "data/processed/consumeriq_hetero_graph.pt",
) -> tuple[HeteroData, dict]:
    """
    Loads serialized PyTorch Geometric HeteroData graph object and metadata from disk.
    Returns: (hetero_data, metadata)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HeteroData graph file not found at: {filepath}")
    
    logger.info(f"Loading PyTorch Geometric HeteroData graph from {filepath}...")
    payload = torch.load(filepath, weights_only=False)
    if isinstance(payload, dict) and "graph" in payload:
        return payload["graph"], payload.get("metadata", {})
    return payload, {}


def run_graph_etl_pipeline(
    input_csv_path: str = "data/processed/unified_consumeriq_real.csv",
    output_graph_path: str = "data/processed/consumeriq_hetero_graph.pt",
) -> tuple[HeteroData, dict]:
    """
    End-to-end orchestration pipeline to parse unified transaction logs into PyG HeteroData.
    """
    logger.info(f"=== Starting ConsumerIQ Graph Neural Network ETL Pipeline ===")
    logger.info(f"Input transaction dataset: {input_csv_path}")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Transaction dataset not found at '{input_csv_path}'.")

    # Read transaction logs with explicit dtypes
    df = pd.read_csv(
        input_csv_path,
        dtype={
            "customer_id": str,
            "product_id": str,
            "quantity": np.int32,
            "unit_price": np.float32,
            "total_sales_value": np.float32,
            "dataset_source": str,
            "product_category": str,
        },
    )
    logger.info(f"Loaded {len(df):,} transaction records.")

    # Clean potential NaN identifiers
    df = df.dropna(subset=["customer_id", "product_id", "quantity", "total_sales_value"]).copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["product_id"] = df["product_id"].astype(str).str.strip()

    # Build HeteroData
    data, mappings = build_hetero_data(df)

    # Persist to disk with index mappings
    save_metadata = {
        "num_records": len(df),
        "customer_to_idx": mappings["customer_to_idx"],
        "product_to_idx": mappings["product_to_idx"],
    }
    save_hetero_graph(data, filepath=output_graph_path, metadata=save_metadata)

    return data, mappings


if __name__ == "__main__":
    run_graph_etl_pipeline()
