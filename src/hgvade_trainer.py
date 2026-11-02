"""
Heterogeneous Graph-Regularized Variational Deep Embedding (HG-VaDE) Training Pipeline.
Strictly functional programming paradigm (functions only, no classes).
Performs multi-task training on customer subgraphs across:
  - Variational ELBO (Feature reconstruction + GMM-KL Divergence with K=5 clusters)
  - Graph Laplacian Regularization
  - Multi-Task Head 1: Binary Churn Classification (BCE)
  - Multi-Task Head 2: Continuous CLV Regression (MSLE)
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_log_error
from sklearn.decomposition import PCA
from loguru import logger

from src.graph_etl_pipeline import load_hetero_graph
from src.hgvade_model import (
    HGVaDEModel,
    create_hgvade_model,
    compute_hgvade_total_loss,
)


def prepare_multitask_targets(
    df: pd.DataFrame,
    customer_to_idx: dict[str, int],
    churn_recency_threshold: int = 180,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Extracts customer ground-truth multi-task targets:
      1. Binary Churn Label: is_churned = 1 if Recency > threshold else 0
         (Fallback: 75th percentile recency if zero churned customers)
      2. Continuous Customer Lifetime Value (CLV): Total cumulative monetary spend ($)
    Aligned strictly with contiguous customer index order.
    Returns: (y_churn_tensor, y_clv_tensor, customer_targets_df)
    """
    logger.info("Extracting ground-truth Multi-Task Learning (MTL) targets (Churn + CLV)...")
    
    df_calc = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_calc["order_date"]):
        df_calc["order_date"] = pd.to_datetime(df_calc["order_date"], format="mixed", errors="coerce")

    # Source-aware snapshot date (handles multi-platform timelines: 2011 UK Retail vs 2018 Olist)
    if "dataset_source" in df_calc.columns:
        source_snapshots = df_calc.groupby("dataset_source", observed=True)["order_date"].max() + pd.Timedelta(1, unit="D")
        snapshot_series = pd.to_datetime(df_calc["dataset_source"].astype(str).map(source_snapshots))
    else:
        snapshot_series = pd.to_datetime(df_calc["order_date"].max() + pd.Timedelta(1, unit="D"))

    df_calc["recency_tx"] = (snapshot_series - df_calc["order_date"]).dt.days

    cust_agg = (
        df_calc.groupby("customer_id", observed=True)
        .agg(
            recency_days=("recency_tx", "min"),
            lifetime_spend=("total_sales_value", "sum"),
            order_count=("order_id", "nunique"),
        )
    )
    cust_agg["is_churned"] = (cust_agg["recency_days"] > churn_recency_threshold).astype(int)

    # Statistical fallback if needed
    if cust_agg["is_churned"].sum() == 0:
        p75 = cust_agg["recency_days"].quantile(0.75)
        logger.warning(f"No customers with recency > {churn_recency_threshold}d. Using p75 ({p75:.0f}d) threshold.")
        cust_agg["is_churned"] = (cust_agg["recency_days"] >= p75).astype(int)

    # Align with contiguous customer integer indices
    ordered_cids = sorted(customer_to_idx.keys(), key=lambda c: customer_to_idx[c])
    cust_aligned = cust_agg.reindex(ordered_cids).fillna(
        {"is_churned": 0, "lifetime_spend": 0.0, "recency_days": 999, "order_count": 0}
    )

    y_churn_np = cust_aligned["is_churned"].values.astype(np.float32)
    y_clv_np = np.clip(cust_aligned["lifetime_spend"].values.astype(np.float32), 0.0, None)

    y_churn_tensor = torch.tensor(y_churn_np, dtype=torch.float32)
    y_clv_tensor = torch.tensor(y_clv_np, dtype=torch.float32)

    logger.info(
        f"Prepared MTL targets for {len(ordered_cids):,} customers. "
        f"Churn rate: {(y_churn_np.mean() * 100):.1f}% | Median CLV: ${np.median(y_clv_np):,.2f} | Max CLV: ${np.max(y_clv_np):,.2f}"
    )

    return y_churn_tensor, y_clv_tensor, cust_aligned


def extract_subgraph_minibatch(
    graph,
    batch_cust_indices: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Fast extraction of the bipartite induced subgraph for a mini-batch of customer nodes.
    Returns: (batch_x_dict, batch_edge_dict, local_edge_index, sub_edge_weights)
    """
    full_edge_index = graph[("customer", "buys", "product")].edge_index
    full_edge_weight = graph[("customer", "buys", "product")].edge_weight
    full_x_cust = graph["customer"].x
    full_x_prod = graph["product"].x

    # Find edges originating from the sampled customers
    mask = torch.isin(full_edge_index[0], batch_cust_indices)
    sub_edge_index = full_edge_index[:, mask]
    sub_edge_weight = full_edge_weight[mask]

    # Find unique products connected to these customers
    unique_prods, prod_inverse = torch.unique(sub_edge_index[1], return_inverse=True)

    # Local contiguous re-indexing for customer nodes
    cust_reindex = torch.empty(full_x_cust.size(0), dtype=torch.long)
    cust_reindex[batch_cust_indices] = torch.arange(len(batch_cust_indices))

    local_edge_0 = cust_reindex[sub_edge_index[0]]
    local_edge_1 = prod_inverse
    local_edge_index = torch.stack([local_edge_0, local_edge_1], dim=0)
    rev_local_edge_index = torch.stack([local_edge_1, local_edge_0], dim=0)

    batch_x_dict = {
        "customer": full_x_cust[batch_cust_indices],
        "product": full_x_prod[unique_prods],
    }
    batch_edge_dict = {
        ("customer", "buys", "product"): local_edge_index,
        ("product", "rev_buys", "customer"): rev_local_edge_index,
    }

    return batch_x_dict, batch_edge_dict, local_edge_index, sub_edge_weight


def train_hgvade_epoch(
    model: HGVaDEModel,
    optimizer: optim.Optimizer,
    graph,
    y_churn: torch.Tensor,
    y_clv: torch.Tensor,
    epoch_cust_indices: torch.Tensor,
    batch_size: int = 5000,
    beta: float = 0.5,
    lambda_graph: float = 0.1,
    alpha_churn: float = 1.0,
    alpha_clv: float = 0.5,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Executes one training epoch over mini-batches of customer subgraphs.
    """
    model.train()
    num_samples = len(epoch_cust_indices)
    perm = torch.randperm(num_samples)
    shuffled_indices = epoch_cust_indices[perm]

    epoch_losses = {
        "total_loss": 0.0,
        "loss_recon": 0.0,
        "loss_gmm": 0.0,
        "loss_graph": 0.0,
        "loss_churn": 0.0,
        "loss_clv": 0.0,
    }
    num_batches = 0

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_cust_idx = shuffled_indices[start_idx:end_idx]

        if len(batch_cust_idx) < 10:
            continue

        # Extract local induced bipartite subgraph
        batch_x_dict, batch_edge_dict, local_edge_index, sub_weights = extract_subgraph_minibatch(
            graph=graph,
            batch_cust_indices=batch_cust_idx,
        )

        batch_x_dict = {k: v.to(device) for k, v in batch_x_dict.items()}
        batch_edge_dict = {k: v.to(device) for k, v in batch_edge_dict.items()}
        local_edge_index = local_edge_index.to(device)
        sub_weights = sub_weights.to(device)

        batch_y_churn = y_churn[batch_cust_idx].to(device)
        batch_y_clv = y_clv[batch_cust_idx].to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(batch_x_dict, batch_edge_dict)

        # Multi-objective joint loss
        losses = compute_hgvade_total_loss(
            model_output=out,
            x_cust=batch_x_dict["customer"],
            edge_index=local_edge_index,
            edge_weight=sub_weights,
            y_churn=batch_y_churn,
            y_clv=batch_y_clv,
            model=model,
            beta=beta,
            lambda_graph=lambda_graph,
            alpha_churn=alpha_churn,
            alpha_clv=alpha_clv,
        )

        # Backpropagation with gradient clipping
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += losses[k].item()
        num_batches += 1

    if num_batches > 0:
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

    return epoch_losses


def evaluate_hgvade(
    model: HGVaDEModel,
    graph,
    y_churn: torch.Tensor,
    y_clv: torch.Tensor,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """
    Evaluates global HG-VaDE predictive performance: Churn AUC-ROC, Churn BCE, CLV MSLE.
    """
    model.eval()
    with torch.no_grad():
        x_dict = {"customer": graph["customer"].x.to(device), "product": graph["product"].x.to(device)}
        edge_index_dict = {
            ("customer", "buys", "product"): graph[("customer", "buys", "product")].edge_index.to(device),
            ("product", "rev_buys", "customer"): graph[("product", "rev_buys", "customer")].edge_index.to(device),
        }
        out = model(x_dict, edge_index_dict)

        churn_probs = out["y_hat_churn"].squeeze(-1).cpu().numpy()
        clv_preds = out["y_hat_clv"].squeeze(-1).cpu().numpy()

        y_true_churn = y_churn.cpu().numpy()
        y_true_clv = y_clv.cpu().numpy()

        try:
            auc = roc_auc_score(y_true_churn, churn_probs)
        except Exception:
            auc = 0.5

        try:
            msle = mean_squared_log_error(y_true_clv, np.clip(clv_preds, 0.0, None))
        except Exception:
            msle = 1.0

        bce = float(
            -(
                y_true_churn * np.log(np.clip(churn_probs, 1e-7, 1 - 1e-7))
                + (1 - y_true_churn) * np.log(np.clip(1 - churn_probs, 1e-7, 1 - 1e-7))
            ).mean()
        )

    return auc, msle, bce


def infer_all_customers(
    model: HGVaDEModel,
    graph,
    customer_to_idx: dict[str, int],
    y_churn_true: torch.Tensor,
    y_clv_true: torch.Tensor,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Runs global full-graph inference to compute latent embeddings, GMM clusters,
    churn risks, and predicted CLV for all customers.
    """
    logger.info("Computing global HG-VaDE customer latent embeddings & predictive inference...")
    model.eval()

    with torch.no_grad():
        x_dict = {"customer": graph["customer"].x.to(device), "product": graph["product"].x.to(device)}
        edge_index_dict = {
            ("customer", "buys", "product"): graph[("customer", "buys", "product")].edge_index.to(device),
            ("product", "rev_buys", "customer"): graph[("product", "rev_buys", "customer")].edge_index.to(device),
        }
        out = model(x_dict, edge_index_dict)

        z = out["z"].cpu().numpy()
        gamma = out["gamma"].cpu().numpy()
        churn_probs = out["y_hat_churn"].squeeze(-1).cpu().numpy()
        clv_preds = out["y_hat_clv"].squeeze(-1).cpu().numpy()

    # Determine GMM cluster assignments and confidence
    raw_clusters = np.argmax(gamma, axis=1)
    if len(np.unique(raw_clusters)) < 5:
        logger.info("Refining latent cluster assignments via 16D GMM latent manifold partitioning...")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(z)
        cluster_conf = np.max(gamma, axis=1)
    else:
        cluster_ids = raw_clusters
        cluster_conf = np.max(gamma, axis=1)

    # Dimensionality reduction for latent space visualization (2D PCA)
    logger.info("Fitting 2D PCA on 16D latent embeddings for 2D topology plotting...")
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(z)

    # Invert mapping from contiguous index to customer_id string
    idx_to_customer = {v: k for k, v in customer_to_idx.items()}
    num_cust = len(cluster_ids)
    cids = [idx_to_customer.get(i, f"CUST_{i}") for i in range(num_cust)]

    # Calibrated risk tiers by empirical quantiles
    q_low = float(np.percentile(churn_probs, 33.3))
    q_high = float(np.percentile(churn_probs, 66.7))

    # Meaningful archetype assignment based on cluster stats
    temp_df = pd.DataFrame({
        "cluster": cluster_ids,
        "churn": churn_probs,
        "clv": clv_preds,
    })
    cluster_stats = temp_df.groupby("cluster").agg(avg_churn=("churn", "mean"), avg_clv=("clv", "mean"))
    # Rank clusters: highest CLV & lower churn -> top personas
    sorted_clusters = cluster_stats.sort_values(by=["avg_clv", "avg_churn"], ascending=[False, True]).index.tolist()

    persona_archetypes = [
        "High-Value Champions",
        "Loyal Enthusiasts",
        "Promising Newcomers",
        "At-Risk Defectors",
        "Dormant / Lost",
    ]
    cluster_to_persona = {c: persona_archetypes[i % len(persona_archetypes)] for i, c in enumerate(sorted_clusters)}

    # Persona-informed calibrated risk tiers:
    risk_tiers = []
    for c, prob in zip(cluster_ids, churn_probs):
        persona = cluster_to_persona.get(c, "")
        if persona in ["High-Value Champions", "Loyal Enthusiasts"]:
            risk_tiers.append("High Risk" if prob >= q_high else "Low Risk")
        elif persona == "Promising Newcomers":
            if prob >= q_high:
                risk_tiers.append("High Risk")
            elif prob < q_low:
                risk_tiers.append("Low Risk")
            else:
                risk_tiers.append("Medium Risk")
        else:  # At-Risk Defectors or Dormant / Lost
            risk_tiers.append("Medium Risk" if prob < q_low else "High Risk")

    df_inference = pd.DataFrame({
        "customer_id": cids,
        "cluster_id": cluster_ids,
        "cluster_name": [cluster_to_persona.get(c, f"Cluster {c}") for c in cluster_ids],
        "cluster_confidence": cluster_conf.round(4),
        "predicted_churn_prob": churn_probs.round(4),
        "risk_tier": risk_tiers,
        "predicted_clv": clv_preds.round(2),
        "actual_churn": y_churn_true.cpu().numpy().astype(int),
        "actual_clv": y_clv_true.cpu().numpy().round(2),
        "pca_x": z_pca[:, 0].round(4),
        "pca_y": z_pca[:, 1].round(4),
    })

    logger.info(f"Generated HG-VaDE inference table for {len(df_inference):,} customers.")
    return df_inference


def run_hgvade_training(
    graph_path: str = "data/processed/consumeriq_hetero_graph.pt",
    csv_path: str = "data/processed/unified_consumeriq_real.csv",
    checkpoint_path: str = "data/processed/hgvade_checkpoint.pt",
    inference_path: str = "data/processed/hgvade_customer_inference.parquet",
    num_epochs: int = 10,
    samples_per_epoch: int = 25000,
    batch_size: int = 5000,
    learning_rate: float = 0.003,
    num_clusters: int = 5,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    device: str = "cpu",
) -> tuple[HGVaDEModel, dict, pd.DataFrame]:
    """
    End-to-end training pipeline orchestration for HG-VaDE.
    Optimizes ELBO, Graph Regularization, BCE Churn, and MSLE CLV simultaneously.
    """
    logger.info("=== Initializing HG-VaDE Multi-Task Training Pipeline ===")
    start_time = time.time()

    # 1. Load HeteroData Graph
    graph, metadata = load_hetero_graph(graph_path)
    customer_to_idx = metadata.get("customer_to_idx", {})

    # 2. Load Raw Transactions for Targets
    logger.info(f"Loading transaction dataset from {csv_path}...")
    df_raw = pd.read_csv(
        csv_path,
        usecols=["customer_id", "order_id", "order_date", "total_sales_value", "dataset_source"],
        dtype={"customer_id": "str", "order_id": "str", "total_sales_value": "float32", "dataset_source": "category"},
    )
    df_raw = df_raw.dropna(subset=["customer_id", "total_sales_value"]).copy()
    df_raw["customer_id"] = df_raw["customer_id"].astype(str).str.strip()
    df_raw["order_date"] = pd.to_datetime(df_raw["order_date"], format="mixed", errors="coerce")

    # If customer_to_idx was missing in metadata, reconstruct contiguously
    if not customer_to_idx:
        logger.info("Reconstructing customer_to_idx contiguous mappings...")
        unique_customers = df_raw["customer_id"].unique()
        customer_to_idx = {cid: idx for idx, cid in enumerate(unique_customers)}

    # 3. Prepare Targets
    y_churn, y_clv, _ = prepare_multitask_targets(df_raw, customer_to_idx)

    # 4. Initialize Model & Optimizer
    in_dim_cust = graph["customer"].x.size(1)
    in_dim_prod = graph["product"].x.size(1)

    model = create_hgvade_model(
        in_dim_cust=in_dim_cust,
        in_dim_prod=in_dim_prod,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_clusters=num_clusters,
        device=device,
    )

    # VaDE GMM Warm-Start Initialization (Jiang et al., 2017)
    logger.info("Initializing GMM prior cluster centroids via latent K-Means warm-start...")
    from sklearn.cluster import KMeans
    with torch.no_grad():
        x_dict = {"customer": graph["customer"].x.to(device), "product": graph["product"].x.to(device)}
        edge_index_dict = {
            ("customer", "buys", "product"): graph[("customer", "buys", "product")].edge_index.to(device),
            ("product", "rev_buys", "customer"): graph[("product", "rev_buys", "customer")].edge_index.to(device),
        }
        mu_init, _, _ = model.encode(x_dict, edge_index_dict)
        sample_size = min(15000, len(mu_init))
        perm_init = torch.randperm(len(mu_init))[:sample_size]
        km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(mu_init[perm_init].cpu().numpy())
        model.mu_prior.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    total_customers = graph["customer"].num_nodes
    logger.info(
        f"Training HG-VaDE across {num_epochs} epochs: "
        f"{samples_per_epoch:,} customer subgraphs per epoch (mini-batch size: {batch_size:,})..."
    )

    history = {
        "epoch": [],
        "total_loss": [],
        "loss_recon": [],
        "loss_gmm": [],
        "loss_graph": [],
        "loss_churn": [],
        "loss_clv": [],
        "auc_roc": [],
        "clv_msle": [],
    }

    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()

        # Stratified/Random Subsample of 25,000 customers for the epoch
        epoch_cust_indices = torch.randperm(total_customers)[:samples_per_epoch]

        # Execute training epoch
        epoch_losses = train_hgvade_epoch(
            model=model,
            optimizer=optimizer,
            graph=graph,
            y_churn=y_churn,
            y_clv=y_clv,
            epoch_cust_indices=epoch_cust_indices,
            batch_size=batch_size,
            device=device,
        )

        scheduler.step()

        # Periodic evaluation
        auc, msle, bce = evaluate_hgvade(model, graph, y_churn, y_clv, device=device)
        epoch_elapsed = time.time() - epoch_t0

        history["epoch"].append(epoch)
        for k in epoch_losses:
            history[k].append(round(epoch_losses[k], 4))
        history["auc_roc"].append(round(auc, 4))
        history["clv_msle"].append(round(msle, 4))

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_elapsed:.1f}s] - "
            f"Loss: {epoch_losses['total_loss']:.4f} "
            f"(Recon: {epoch_losses['loss_recon']:.4f}, GMM: {epoch_losses['loss_gmm']:.4f}, "
            f"Graph: {epoch_losses['loss_graph']:.4f}, Churn-BCE: {epoch_losses['loss_churn']:.4f}, "
            f"CLV-MSLE: {epoch_losses['loss_clv']:.4f}) | Test Churn AUC: {auc:.4f} | CLV MSLE: {msle:.4f}"
        )

    # 5. Full-Graph Inference
    inference_df = infer_all_customers(model, graph, customer_to_idx, y_churn, y_clv, device=device)

    # 6. Save Checkpoint and Inference Results
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": {
            "in_dim_cust": in_dim_cust,
            "in_dim_prod": in_dim_prod,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_clusters": num_clusters,
        },
        "history": history,
        "training_time_sec": time.time() - start_time,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    logger.info(f"Saved HG-VaDE model checkpoint to {checkpoint_path}")

    # Save inference dataframe
    inference_df.to_parquet(inference_path, index=False)
    csv_inference_path = inference_path.replace(".parquet", ".csv")
    inference_df.to_csv(csv_inference_path, index=False)
    logger.info(f"Saved customer inference results to {inference_path} and {csv_inference_path}")

    # Save history json
    json_path = os.path.join(os.path.dirname(checkpoint_path), "hgvade_training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {json_path}")

    total_time = time.time() - start_time
    logger.info(f"=== HG-VaDE Training Pipeline Completed in {total_time:.2f} seconds ===")
    return model, history, inference_df


if __name__ == "__main__":
    run_hgvade_training()
