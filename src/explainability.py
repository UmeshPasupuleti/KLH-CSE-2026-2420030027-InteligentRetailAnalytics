"""
Explainable AI (XAI) Module using SHAP DeepExplainer for ConsumerIQ.
Conditioned on HG-VaDE Gaussian Mixture Model (GMM) Latent Clusters.
Strictly functional programming paradigm (functions only, no unnecessary classes).

Architecture & Pipeline:
  1. Deep Neural Network (CustomerChurnDeepNet) mapping customer behavioral features to churn risk.
  2. SHAP DeepExplainer computing DeepLIFT Shapley values on the learned representations.
  3. GMM Cluster Partitioning: Explanations conditioned across the 5 HG-VaDE probabilistic personas:
     - Cluster 0: High-Value Champions
     - Cluster 1: At-Risk Defectors
     - Cluster 2: Promising Newcomers
     - Cluster 3: Dormant / Lost
     - Cluster 4: Loyal Enthusiasts
  4. Publication-grade segment-conditioned beeswarm summary visualizations and feature importance bars.
"""

import os
import pickle
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

# Gracefully handle SHAP import in constrained environments
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP package imported successfully.")
except (ImportError, OSError) as _shap_err:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    logger.warning(f"SHAP could not be imported: {_shap_err}. XAI will run in graceful fallback mode.")


# ============================================================================
# PyTorch Deep Neural Architecture for DeepExplainer
# ============================================================================


class CustomerChurnDeepNet(nn.Module):
    """
    Feedforward Deep Neural Network predicting customer churn probability
    from normalized customer behavioral graph features.
    Designed for SHAP DeepExplainer DeepLIFT gradient backpropagation.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classifier heads."""
        z = self.encoder(x)
        return self.classifier(z)


# Feature names mapped to customer node feature indices
FEATURE_NAMES: list[str] = [
    "Total Order Volume",
    "Total Monetary Spend ($)",
    "Catalog Product Breadth",
    "Average Item Spend ($)",
]

CLUSTER_METADATA: dict[int, dict[str, str]] = {
    0: {
        "name": "High-Value Champions",
        "color": "#3B82F6",
        "badge": "Champion",
        "strategy": "VIP concierge rewards, early product access, and executive retention incentives.",
    },
    1: {
        "name": "At-Risk Defectors",
        "color": "#F59E0B",
        "badge": "At Risk",
        "strategy": "Urgent automated win-back discounts, push re-engagement campaigns, and price-match guarantees.",
    },
    2: {
        "name": "Promising Newcomers",
        "color": "#8B5CF6",
        "badge": "Newcomer",
        "strategy": "Seamless 2nd-purchase nurturing, curated category cross-sell, and loyalty points onboarding.",
    },
    3: {
        "name": "Dormant / Lost",
        "color": "#EF4444",
        "badge": "Dormant",
        "strategy": "Deep clearance reactivation promos, re-subscription triggers, or automated sunset protocols.",
    },
    4: {
        "name": "Loyal Enthusiasts",
        "color": "#10B981",
        "badge": "Loyalist",
        "strategy": "Referral advocacy bonuses, cross-category bundle discounts, and community recognition.",
    },
}

CACHE_FILE_PATH: str = os.path.join("data", "processed", "gmm_cluster_shap_results.pkl")


# ============================================================================
# Pure Functional Deep Neural Model Pipeline
# ============================================================================


def build_deep_churn_network(
    in_dim: int = 4,
    hidden_dim: int = 32,
    latent_dim: int = 16,
) -> CustomerChurnDeepNet:
    """
    Factory function initializing the CustomerChurnDeepNet module.
    """
    return CustomerChurnDeepNet(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)


def train_deep_churn_network(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 12,
    lr: float = 0.008,
    batch_size: int = 1024,
    device: str = "cpu",
) -> CustomerChurnDeepNet:
    """
    Trains CustomerChurnDeepNet on customer features and targets via mini-batch SGD (Adam).
    """
    logger.info(f"Training CustomerChurnDeepNet on {len(X_train):,} samples across {epochs} epochs...")
    model = build_deep_churn_network(in_dim=X_train.shape[1]).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_x, b_y in loader:
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_x)
        epoch_loss /= len(X_train)

    logger.info(f"CustomerChurnDeepNet trained successfully. Final MSE Loss: {epoch_loss:.5f}")
    model.eval()
    return model


# ============================================================================
# SHAP DeepExplainer Segment-Conditioned Pipeline
# ============================================================================


def compute_gmm_cluster_shap_explanations(
    model: nn.Module,
    X_cust: torch.Tensor,
    df_inf: pd.DataFrame,
    sample_per_cluster: int = 150,
    bg_samples: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Runs SHAP DeepExplainer across each of the 5 HG-VaDE GMM clusters separately.
    Returns dictionary with per-cluster SHAP matrices, feature values, and metadata.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available; returning empty explanation dictionary.")
        return {}

    logger.info(f"Initializing SHAP DeepExplainer with {bg_samples} background customer vectors...")
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Global background reference distribution
    num_total = len(X_cust)
    bg_indices = np.random.choice(num_total, size=min(bg_samples, num_total), replace=False)
    bg_tensor = X_cust[torch.tensor(bg_indices, dtype=torch.long)]

    explainer = shap.DeepExplainer(model, bg_tensor)

    cluster_ids = sorted(df_inf["cluster_id"].unique())
    results: dict = {}

    for cid in cluster_ids:
        c_meta = CLUSTER_METADATA.get(int(cid), {
            "name": f"Cluster {cid}",
            "color": "#6366F1",
            "badge": f"C{cid}",
            "strategy": "Tailored customer retention actions.",
        })
        c_name = c_meta["name"]

        # Filter indices of customers belonging to this GMM cluster
        c_indices = np.where(df_inf["cluster_id"] == cid)[0]
        total_in_cluster = len(c_indices)

        if total_in_cluster == 0:
            continue

        cohort_size = min(sample_per_cluster, total_in_cluster)
        sampled_c_idx = np.random.choice(c_indices, size=cohort_size, replace=False)

        # Extract tensor for this cluster cohort
        c_tensor = X_cust[torch.tensor(sampled_c_idx, dtype=torch.long)]

        logger.info(f"Computing DeepLIFT SHAP values for GMM Cluster {cid} ({c_name}, n={cohort_size})...")
        shap_vals = explainer.shap_values(c_tensor)

        # Handle tensor dimensionalities
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals_2d = shap_vals.squeeze(-1)
        elif isinstance(shap_vals, list) and len(shap_vals) > 0:
            shap_vals_2d = shap_vals[0]
            if shap_vals_2d.ndim == 3:
                shap_vals_2d = shap_vals_2d.squeeze(-1)
        else:
            shap_vals_2d = np.array(shap_vals)

        # Construct dataframe of feature values for summary beeswarm
        features_sample = c_tensor.cpu().numpy()
        features_df = pd.DataFrame(features_sample, columns=FEATURE_NAMES)

        # Calculate mean absolute SHAP values (|SHAP|)
        mean_abs_shap = np.mean(np.abs(shap_vals_2d), axis=0)
        feat_importance = dict(zip(FEATURE_NAMES, [float(v) for v in mean_abs_shap]))
        top_driver_name = max(feat_importance, key=feat_importance.get)  # type: ignore[arg-type]

        # Calculate cluster baseline metrics
        c_sub_df = df_inf.iloc[sampled_c_idx]
        mean_churn = float(c_sub_df["predicted_churn_prob"].mean())
        mean_clv = float(c_sub_df["predicted_clv"].mean())

        results[int(cid)] = {
            "cluster_id": int(cid),
            "cluster_name": c_name,
            "cluster_color": c_meta["color"],
            "cluster_badge": c_meta["badge"],
            "total_customers": total_in_cluster,
            "cohort_size": cohort_size,
            "mean_churn_prob": round(mean_churn, 4),
            "mean_clv": round(mean_clv, 2),
            "top_churn_driver": top_driver_name,
            "top_driver_val": round(feat_importance[top_driver_name], 4),
            "strategy": c_meta["strategy"],
            "shap_values": shap_vals_2d,
            "features_df": features_df,
            "feature_importance": feat_importance,
        }

    return results


# ============================================================================
# Publication-Grade Visualization Functions
# ============================================================================


def generate_cluster_beeswarm_plot(
    cluster_shap_data: dict,
    max_display: int = 4,
) -> Optional[plt.Figure]:
    """
    Renders a standalone SHAP Beeswarm Summary Plot for a single GMM cluster.
    Styled with the ConsumerIQ dark theme (#0F172A slate).
    """
    if not SHAP_AVAILABLE or not cluster_shap_data:
        return None

    shap_vals = cluster_shap_data["shap_values"]
    features_df = cluster_shap_data["features_df"]
    c_name = cluster_shap_data["cluster_name"]
    cid = cluster_shap_data["cluster_id"]
    n_samples = cluster_shap_data["cohort_size"]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=130)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    plt.rcParams["text.color"] = "#F8FAFC"
    plt.rcParams["axes.labelcolor"] = "#F8FAFC"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["font.family"] = "sans-serif"

    shap.summary_plot(
        shap_vals,
        features_df,
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    plt.title(
        f"GMM Cluster {cid}: {c_name} — SHAP DeepExplainer Summary (n={n_samples})",
        color="#F8FAFC",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("SHAP Attribution Value (Impact on Churn Log-Odds / Risk)", color="#94A3B8", fontsize=9.5)
    plt.tight_layout()
    return fig


def generate_cluster_feature_importance_bar(cluster_shap_data: dict) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of mean absolute SHAP values for the cluster.
    """
    feat_imp = cluster_shap_data["feature_importance"]
    c_color = cluster_shap_data["cluster_color"]
    c_name = cluster_shap_data["cluster_name"]

    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    df_plot = pd.DataFrame(sorted_feats, columns=["Feature", "MeanAbsoluteSHAP"])
    df_plot = df_plot.sort_values(by="MeanAbsoluteSHAP", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["MeanAbsoluteSHAP"],
            y=df_plot["Feature"],
            orientation="h",
            marker=dict(
                color=c_color,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            text=df_plot["MeanAbsoluteSHAP"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"<b>Mean Feature Impact (|SHAP|) — {c_name}</b>",
        height=320,
        margin=dict(t=40, b=20, l=140, r=40),
        xaxis=dict(
            title="Mean Absolute SHAP Value",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
        ),
        yaxis=dict(
            showgrid=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
    )
    return fig


def generate_multi_cluster_comparison_chart(clusters_dict: dict) -> go.Figure:
    """
    Renders a grouped comparison bar chart showing mean absolute SHAP feature attributions
    across all 5 GMM clusters side-by-side.
    """
    records = []
    for cid, data in clusters_dict.items():
        for feat, val in data["feature_importance"].items():
            records.append({
                "Cluster": f"{data['cluster_name']} (C{cid})",
                "Feature": feat,
                "Mean |SHAP| Attribution": val,
            })

    df_comp = pd.DataFrame(records)

    fig = px.bar(
        df_comp,
        x="Feature",
        y="Mean |SHAP| Attribution",
        color="Cluster",
        barmode="group",
        title="<b>Cross-Cluster Feature Sensitivity Comparison (SHAP DeepExplainer)</b>",
        color_discrete_map={
            f"{v['cluster_name']} (C{k})": v["cluster_color"] for k, v in clusters_dict.items()
        },
    )

    fig.update_layout(
        height=360,
        margin=dict(t=40, b=30, l=20, r=20),
        xaxis_title="",
        yaxis_title="Mean |SHAP| Attribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="",
        ),
    )
    return fig


# ============================================================================
# Caching & Master Orchestration Pipeline
# ============================================================================


def save_gmm_cluster_shap_cache(
    results: dict,
    cache_path: str = CACHE_FILE_PATH,
) -> None:
    """
    Serializes cluster-conditioned SHAP results to disk.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved GMM cluster SHAP cache to {cache_path}")


def load_cached_gmm_cluster_shap(
    cache_path: str = CACHE_FILE_PATH,
) -> Optional[dict]:
    """
    Loads pre-computed GMM cluster SHAP results if cache file exists.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded cached GMM cluster SHAP results from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to read SHAP cache at {cache_path}: {e}")
    return None


def run_cluster_conditioned_shap_pipeline(
    sample_per_cluster: int = 150,
    force_recompute: bool = False,
) -> dict:
    """
    Master pipeline orchestrating:
      1. Loading graph customer features and HG-VaDE GMM inference table.
      2. Training CustomerChurnDeepNet surrogate network.
      3. Executing SHAP DeepExplainer on the 5 GMM clusters.
      4. Computing segment summaries and serializing to disk.
    """
    if not force_recompute:
        cached = load_cached_gmm_cluster_shap()
        if cached is not None:
            return cached

    logger.info("Running complete GMM Cluster SHAP DeepExplainer pipeline...")
    # Load HeteroData graph and HG-VaDE inference table
    from src.graph_etl_pipeline import load_hetero_graph

    graph, _ = load_hetero_graph()
    inference_path = os.path.join("data", "processed", "hgvade_customer_inference.parquet")
    if not os.path.exists(inference_path):
        inference_path = os.path.join("data", "processed", "hgvade_customer_inference.csv")
        df_inf = pd.read_csv(inference_path)
    else:
        df_inf = pd.read_parquet(inference_path)

    X_cust = graph["customer"].x.float()
    y_churn = torch.tensor(df_inf["predicted_churn_prob"].values, dtype=torch.float32).unsqueeze(-1)

    # Train deep neural network
    deep_net = train_deep_churn_network(X_cust, y_churn, epochs=12, batch_size=1024)

    # Run DeepExplainer across the 5 GMM clusters
    cluster_results = compute_gmm_cluster_shap_explanations(
        model=deep_net,
        X_cust=X_cust,
        df_inf=df_inf,
        sample_per_cluster=sample_per_cluster,
    )

    # Construct executive summary table
    summary_rows = []
    for cid in sorted(cluster_results.keys()):
        data = cluster_results[cid]
        summary_rows.append({
            "Cluster ID": cid,
            "GMM Persona": data["cluster_name"],
            "Customer Count": f"{data['total_customers']:,}",
            "Mean Churn Risk": f"{data['mean_churn_prob'] * 100:.1f}%",
            "Mean CLV ($)": f"${data['mean_clv']:,.2f}",
            "Primary Churn Driver": data["top_churn_driver"],
            "Driver Attribution": f"{data['top_driver_val']:.4f}",
            "Recommended Strategic Lever": data["strategy"],
        })

    payload = {
        "clusters": cluster_results,
        "summary_table": summary_rows,
        "feature_names": FEATURE_NAMES,
        "timestamp": datetime.now().isoformat(),
    }

    save_gmm_cluster_shap_cache(payload)
    return payload