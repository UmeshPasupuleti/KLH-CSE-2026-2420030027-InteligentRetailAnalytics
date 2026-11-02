"""
Interactive PyVis Subgraph Visualizer for ConsumerIQ Heterogeneous Graph Analytics.
Strictly functional programming paradigm (functions only, no classes).
Generates interactive, physics-enabled HTML network representations of
customer-product bipartite graphs, GMM cluster topologies, and churn risk neighborhoods.
"""

import json
import networkx as nx
import pandas as pd
import numpy as np
import torch
from pyvis.network import Network
from loguru import logger


# Color palettes for intuitive visualization
PERSONA_COLORS = {
    "High-Value Champions": "#3B82F6",  # Vibrant Blue
    "Loyal Enthusiasts": "#10B981",     # Emerald Green
    "Promising Newcomers": "#8B5CF6",   # Purple
    "At-Risk Defectors": "#F59E0B",     # Amber Orange
    "Dormant / Lost": "#EF4444",        # Crimson Red
}

CLUSTER_COLORS = {
    0: "#EF4444",  # Dormant / Lost
    1: "#10B981",  # Loyal Enthusiasts
    2: "#8B5CF6",  # Promising Newcomers
    3: "#F59E0B",  # At-Risk Defectors
    4: "#3B82F6",  # High-Value Champions
}

RISK_COLORS = {
    "Low Risk": "#10B981",     # Emerald Green
    "Medium Risk": "#F59E0B",  # Amber Orange
    "High Risk": "#EF4444",    # Crimson Red
}

PRODUCT_COLOR = "#0284C7"      # Deep Sky Blue
FOCAL_CUST_COLOR = "#EC4899"   # Hot Pink / Magenta for Focal Node


def build_customer_neighborhood_networkx(
    graph,
    target_cust_id: str,
    customer_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
    df_inference: pd.DataFrame = None,
    df_raw: pd.DataFrame = None,
    max_products: int = 10,
    max_copurchasers_per_prod: int = 3,
) -> nx.Graph:
    """
    Constructs a NetworkX bipartite graph around a focal customer:
      - Focal Customer Node
      - Connected Product Nodes (top-N by purchase weight)
      - Co-purchasing Customer Neighbors (sharing products with the focal customer)
    """
    G = nx.Graph()

    # Invert product mapping for fast lookup
    idx_to_prod = {v: k for k, v in product_to_idx.items()}
    idx_to_cust = {v: k for k, v in customer_to_idx.items()}

    if target_cust_id not in customer_to_idx:
        logger.warning(f"Customer '{target_cust_id}' not found in graph mapping.")
        return G

    target_idx = customer_to_idx[target_cust_id]
    edge_index = graph[("customer", "buys", "product")].edge_index
    edge_weight = graph[("customer", "buys", "product")].edge_weight

    # Find edges of the focal customer
    cust_mask = edge_index[0] == target_idx
    focal_edges = edge_index[:, cust_mask]
    focal_weights = edge_weight[cust_mask]

    # Sort products by purchase weight descending
    if len(focal_weights) > 0:
        sorted_indices = torch.argsort(focal_weights, descending=True)
        top_prod_indices = focal_edges[1, sorted_indices][:max_products].tolist()
        top_weights = focal_weights[sorted_indices][:max_products].tolist()
    else:
        top_prod_indices = []
        top_weights = []

    # Fast metadata lookup dictionaries
    inf_lookup = {}
    if df_inference is not None and not df_inference.empty:
        inf_lookup = df_inference.set_index("customer_id").to_dict("index")

    # Add Focal Customer Node
    focal_meta = inf_lookup.get(target_cust_id, {})
    focal_cluster = focal_meta.get("cluster_name", "Unknown Cluster")
    focal_churn = focal_meta.get("predicted_churn_prob", 0.0)
    focal_clv = focal_meta.get("predicted_clv", 0.0)

    focal_title = (
        f"<b>Customer (Focal Node):</b> {target_cust_id}<br>"
        f"<b>Persona:</b> {focal_cluster}<br>"
        f"<b>Predicted Churn Risk:</b> {focal_churn * 100:.1f}%<br>"
        f"<b>Predicted CLV:</b> ${focal_clv:,.2f}<br>"
        f"<b>Connected Products:</b> {len(top_prod_indices)}"
    )

    G.add_node(
        f"C_{target_cust_id}",
        label=f"Cust {target_cust_id[-6:]}",
        title=focal_title,
        color=FOCAL_CUST_COLOR,
        size=30,
        shape="star",
        borderWidth=3,
        borderWidthSelected=5,
        font={"color": "#FFFFFF", "size": 14, "bold": True},
    )

    # Product category map if available
    prod_cat_map = {}
    if df_raw is not None and "product_category" in df_raw.columns:
        prod_cat_map = df_raw.groupby("product_id")["product_category"].first().to_dict()

    # Add Connected Products and Edges
    for p_idx, weight in zip(top_prod_indices, top_weights):
        p_id = idx_to_prod.get(p_idx, f"P_{p_idx}")
        p_cat = prod_cat_map.get(p_id, "Catalog Item")
        p_node_id = f"P_{p_id}"

        p_title = (
            f"<b>Product ID:</b> {p_id}<br>"
            f"<b>Category:</b> {p_cat}<br>"
            f"<b>Affinity Weight:</b> {weight:.3f}"
        )

        G.add_node(
            p_node_id,
            label=f"Prod {p_id[-6:]}",
            title=p_title,
            color=PRODUCT_COLOR,
            size=18,
            shape="box",
            font={"color": "#FFFFFF", "size": 11},
        )

        edge_title = f"Affinity: {weight:.2f}"
        G.add_edge(
            f"C_{target_cust_id}",
            p_node_id,
            value=max(1.0, float(weight * 6.0)),
            title=edge_title,
            color={"color": "#38BDF8", "opacity": 0.8},
        )

        # Find Co-Purchasers for this product
        p_mask = (edge_index[1] == p_idx) & (edge_index[0] != target_idx)
        co_cust_indices = edge_index[0, p_mask][:max_copurchasers_per_prod].tolist()
        co_weights = edge_weight[p_mask][:max_copurchasers_per_prod].tolist()

        for c_co_idx, co_w in zip(co_cust_indices, co_weights):
            c_co_id = idx_to_cust.get(c_co_idx, f"C_{c_co_idx}")
            c_co_node_id = f"C_{c_co_id}"

            if not G.has_node(c_co_node_id):
                co_meta = inf_lookup.get(c_co_id, {})
                co_cluster = co_meta.get("cluster_name", "Customer")
                co_risk = co_meta.get("risk_tier", "Low Risk")
                co_churn = co_meta.get("predicted_churn_prob", 0.0)
                co_clv = co_meta.get("predicted_clv", 0.0)
                node_color = RISK_COLORS.get(co_risk, "#10B981")

                co_title = (
                    f"<b>Co-Shopper:</b> {c_co_id}<br>"
                    f"<b>Persona:</b> {co_cluster}<br>"
                    f"<b>Risk Tier:</b> {co_risk} ({co_churn * 100:.1f}%)<br>"
                    f"<b>Predicted CLV:</b> ${co_clv:,.2f}"
                )

                G.add_node(
                    c_co_node_id,
                    label=f"Cust {c_co_id[-6:]}",
                    title=co_title,
                    color=node_color,
                    size=16,
                    shape="dot",
                    font={"color": "#E2E8F0", "size": 10},
                )

            G.add_edge(
                c_co_node_id,
                p_node_id,
                value=max(0.8, float(co_w * 4.0)),
                title=f"Co-Purchase Weight: {co_w:.2f}",
                color={"color": "#64748B", "opacity": 0.5},
            )

    return G


def build_cluster_topology_networkx(
    graph,
    cluster_id: int,
    df_inference: pd.DataFrame,
    customer_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
    df_raw: pd.DataFrame = None,
    max_customers: int = 15,
    max_products: int = 20,
) -> nx.Graph:
    """
    Constructs a NetworkX bipartite graph representing the collective purchase
    topology of an HG-VaDE GMM cluster.
    """
    G = nx.Graph()
    idx_to_prod = {v: k for k, v in product_to_idx.items()}

    # Filter top-confidence customers in this cluster
    cluster_df = df_inference[df_inference["cluster_id"] == cluster_id].sort_values(
        by="cluster_confidence", ascending=False
    ).head(max_customers)

    cluster_name = cluster_df["cluster_name"].iloc[0] if not cluster_df.empty else f"Cluster {cluster_id}"
    cluster_color = PERSONA_COLORS.get(cluster_name, CLUSTER_COLORS.get(cluster_id, "#3B82F6"))

    edge_index = graph[("customer", "buys", "product")].edge_index
    edge_weight = graph[("customer", "buys", "product")].edge_weight

    # Add customer nodes
    cust_indices = []
    for _, row in cluster_df.iterrows():
        cid = str(row["customer_id"])
        if cid in customer_to_idx:
            c_idx = customer_to_idx[cid]
            cust_indices.append(c_idx)

            title = (
                f"<b>Customer:</b> {cid}<br>"
                f"<b>Cluster:</b> {cluster_name}<br>"
                f"<b>Confidence:</b> {row['cluster_confidence'] * 100:.1f}%<br>"
                f"<b>Predicted Churn:</b> {row['predicted_churn_prob'] * 100:.1f}%<br>"
                f"<b>Predicted CLV:</b> ${row['predicted_clv']:,.2f}"
            )

            G.add_node(
                f"C_{cid}",
                label=f"Cust {cid[-5:]}",
                title=title,
                color=cluster_color,
                size=20,
                shape="dot",
                font={"color": "#FFFFFF", "size": 11},
            )

    if not cust_indices:
        return G

    # Find connected products
    cust_tensor = torch.tensor(cust_indices, dtype=torch.long)
    mask = torch.isin(edge_index[0], cust_tensor)
    sub_edge_index = edge_index[:, mask]
    sub_edge_weight = edge_weight[mask]

    # Group by product to find most popular items in this cluster
    prods, counts = torch.unique(sub_edge_index[1], return_counts=True)
    top_p_order = torch.argsort(counts, descending=True)[:max_products]
    top_p_indices = prods[top_p_order].tolist()

    # Product category map
    prod_cat_map = {}
    if df_raw is not None and "product_category" in df_raw.columns:
        prod_cat_map = df_raw.groupby("product_id")["product_category"].first().to_dict()

    for p_idx in top_p_indices:
        pid = idx_to_prod.get(p_idx, f"P_{p_idx}")
        cat = prod_cat_map.get(pid, "Catalog Item")
        p_node_id = f"P_{pid}"

        G.add_node(
            p_node_id,
            label=f"Prod {pid[-5:]}",
            title=f"<b>Product:</b> {pid}<br><b>Category:</b> {cat}",
            color=PRODUCT_COLOR,
            size=18,
            shape="box",
            font={"color": "#FFFFFF", "size": 10},
        )

    # Add edges between customers and top products
    for i in range(sub_edge_index.size(1)):
        u = sub_edge_index[0, i].item()
        v = sub_edge_index[1, i].item()
        w = sub_edge_weight[i].item()

        if v in top_p_indices:
            idx_to_cust = {val: k for k, val in customer_to_idx.items()}
            cid = idx_to_cust.get(u, f"C_{u}")
            pid = idx_to_prod.get(v, f"P_{v}")

            G.add_edge(
                f"C_{cid}",
                f"P_{pid}",
                value=max(1.0, float(w * 5.0)),
                title=f"Weight: {w:.2f}",
                color={"color": "#94A3B8", "opacity": 0.6},
            )

    return G


def generate_pyvis_html(
    G: nx.Graph,
    height: str = "660px",
    width: str = "100%",
    bgcolor: str = "#0F172A",
    font_color: str = "#F8FAFC",
    enable_physics: bool = True,
) -> str:
    """
    Renders a NetworkX graph into an interactive PyVis HTML string with
    responsive canvas styling, hover tooltips, and Barnes-Hut physics stabilization.
    """
    if G.number_of_nodes() == 0:
        return "<div style='color: #94A3B8; text-align: center; padding: 40px;'>No graph nodes found for the selected criteria.</div>"

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=False,
        notebook=False,
    )

    # Ingest NetworkX graph
    net.from_nx(G)

    # Configure physics engine and interactions
    options = {
        "nodes": {
            "borderWidth": 1.5,
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8, "x": 3, "y": 3},
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.2},
            "shadow": {"enabled": False},
        },
        "physics": {
            "enabled": enable_physics,
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 0.5,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": True, "iterations": 150},
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "dragNodes": True,
            "dragView": True,
        },
    }

    net.set_options(json.dumps(options))
    html_content = net.generate_html()

    return html_content
