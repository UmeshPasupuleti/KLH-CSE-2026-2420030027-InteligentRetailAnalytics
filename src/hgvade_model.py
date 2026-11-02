"""
Heterogeneous Graph-Regularized Variational Deep Embedding (HG-VaDE) Model for ConsumerIQ.
Strictly functional programming paradigm (functions only for pipeline and inference routines).
Compresses bipartite HeteroData graphs into a Gaussian Mixture Model (GMM) latent manifold.
Incorporates dual Multi-Task Learning (MTL) heads:
  1. Binary Churn Classification (Binary Cross-Entropy Loss)
  2. Continuous Customer Lifetime Value (CLV) Regression (Mean Squared Logarithmic Error)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from loguru import logger


class HGVaDEModel(nn.Module):
    """
    Heterogeneous Graph-Regularized Variational Deep Embedding (HG-VaDE) Neural Network.
    
    Architecture:
      - Heterogeneous GraphSAGE Encoder (2 layers across bipartite relations)
      - Variational Latent Space (mu_z, logvar_z)
      - Gaussian Mixture Model (GMM) Latent Prior (K components with learnable pi, mu_c, logvar_c)
      - Customer Feature Decoder (Reconstruction)
      - Product Latent Projector (Graph Laplacian Regularization)
      - MTL Head 1: Binary Churn Classifier (BCE)
      - MTL Head 2: Continuous CLV Regressor (MSLE)
    """

    def __init__(
        self,
        in_dim_cust: int = 4,
        in_dim_prod: int = 4,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_clusters: int = 5,
    ) -> None:
        super().__init__()
        self.in_dim_cust = in_dim_cust
        self.in_dim_prod = in_dim_prod
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        # 1. Heterogeneous GraphSAGE Encoder - Layer 1
        self.conv1 = HeteroConv(
            {
                ("customer", "buys", "product"): SAGEConv((in_dim_cust, in_dim_prod), hidden_dim),
                ("product", "rev_buys", "customer"): SAGEConv((in_dim_prod, in_dim_cust), hidden_dim),
            },
            aggr="sum",
        )

        # 2. Heterogeneous GraphSAGE Encoder - Layer 2
        self.conv2 = HeteroConv(
            {
                ("customer", "buys", "product"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("product", "rev_buys", "customer"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            },
            aggr="sum",
        )

        # 3. Customer Variational Latent Heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 4. GMM Prior Parameters (K Gaussian clusters)
        # Cluster mixing weights logits (Softmax -> pi_c)
        self.pi_logits = nn.Parameter(torch.zeros(num_clusters))
        # Cluster centroid means in latent space
        self.mu_prior = nn.Parameter(torch.randn(num_clusters, latent_dim) * 0.1)
        # Cluster log-variances in latent space
        self.logvar_prior = nn.Parameter(torch.zeros(num_clusters, latent_dim))

        # 5. Customer Feature Reconstruction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim_cust),
        )

        # 6. Product Latent Space Projector (for Graph Regularization)
        self.proj_prod = nn.Linear(hidden_dim, latent_dim)

        # 7. Multi-Task Learning Head 1: Binary Churn Classifier
        self.churn_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 8. Multi-Task Learning Head 2: Continuous CLV Regressor (predicts log1p monetary value)
        self.clv_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize bias to empirical log1p median spend (~$95 => ~4.56)
        nn.init.constant_(self.clv_head[-1].bias, 4.5)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the heterogeneous graph using 2-layer GraphSAGE message passing.
        Returns: (mu_z, logvar_z, h_prod)
        """
        # Layer 1
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {k: F.relu(v) for k, v in h1.items()}

        # Layer 2
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: F.relu(v) for k, v in h2.items()}

        h_cust = h2["customer"]
        h_prod = h2["product"]

        mu_z = self.fc_mu(h_cust)
        logvar_z = torch.clamp(self.fc_logvar(h_cust), min=-10.0, max=10.0)

        return mu_z, logvar_z, h_prod

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample z from N(mu, diag(exp(logvar))).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def compute_gamma(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the variational posterior cluster responsibilities gamma_{ic} = q(c | z_i)
        using Bayes' rule over the GMM prior components in log-space for numerical stability.
        Returns:
          gamma: Softmax cluster probabilities of shape (N, K)
          log_p_z_c: Log-joint probability tensor of shape (N, K)
        """
        pi = F.softmax(self.pi_logits, dim=0)  # (K,)
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (K,)

        # Expand z: (N, 1, D)
        z_exp = z.unsqueeze(1)
        # Expand prior parameters: (1, K, D)
        mu_c = self.mu_prior.unsqueeze(0)
        logvar_c = torch.clamp(self.logvar_prior.unsqueeze(0), min=-10.0, max=10.0)
        var_c = torch.exp(logvar_c)

        # log N(z | mu_c, Sigma_c) = -0.5 * sum_d [ log(2*pi) + logvar_c + (z - mu_c)^2 / var_c ]
        d = self.latent_dim
        diff_sq = (z_exp - mu_c) ** 2
        log_norm = -0.5 * (d * math.log(2.0 * math.pi) + torch.sum(logvar_c + diff_sq / (var_c + 1e-8), dim=-1))  # (N, K)

        # log p(z, c) = log pi_c + log p(z | c)
        log_p_z_c = log_pi.unsqueeze(0) + log_norm  # (N, K)

        # gamma_{ic} = Softmax_c(log p(z, c))
        gamma = F.softmax(log_p_z_c, dim=-1)  # (N, K)

        return gamma, log_p_z_c

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through HG-VaDE.
        Returns dictionary of all intermediate and final outputs.
        """
        mu_z, logvar_z, h_prod = self.encode(x_dict, edge_index_dict)
        z = self.reparameterize(mu_z, logvar_z)

        # Cluster responsibilities
        gamma, log_p_z_c = self.compute_gamma(z)

        # Feature reconstruction
        x_hat_cust = self.decoder(z)

        # MTL Predictions
        y_hat_churn = self.churn_head(z)
        y_hat_log_clv = self.clv_head(z)
        y_hat_clv = torch.expm1(torch.clamp(y_hat_log_clv, min=0.0, max=15.0))

        # Product latent projection for graph regularization
        z_prod_proj = self.proj_prod(h_prod)

        return {
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "gamma": gamma,
            "log_p_z_c": log_p_z_c,
            "x_hat_cust": x_hat_cust,
            "y_hat_churn": y_hat_churn,
            "y_hat_log_clv": y_hat_log_clv,
            "y_hat_clv": y_hat_clv,
            "z_prod_proj": z_prod_proj,
            "h_prod": h_prod,
        }


# ============================================================================
# Pure Functional Loss & Training Utilities
# ============================================================================


def create_hgvade_model(
    in_dim_cust: int = 4,
    in_dim_prod: int = 4,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    num_clusters: int = 5,
    device: str = "cpu",
) -> HGVaDEModel:
    """
    Factory function to initialize and configure an HGVaDEModel instance.
    """
    logger.info(
        f"Initializing HGVaDEModel (cust_dim={in_dim_cust}, prod_dim={in_dim_prod}, "
        f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, K={num_clusters}) on {device}"
    )
    model = HGVaDEModel(
        in_dim_cust=in_dim_cust,
        in_dim_prod=in_dim_prod,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_clusters=num_clusters,
    ).to(device)
    return model


def compute_elbo_and_gmm_loss(
    x_cust: torch.Tensor,
    x_hat_cust: torch.Tensor,
    mu_z: torch.Tensor,
    logvar_z: torch.Tensor,
    gamma: torch.Tensor,
    model: HGVaDEModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the VaDE ELBO components:
      1. Reconstruction Loss: MSE on customer normalized features
      2. GMM-KL Divergence: D_KL(q(z, c | x) || p(z, c))
    """
    # 1. Feature reconstruction loss
    loss_recon = F.mse_loss(x_hat_cust, x_cust, reduction="mean")

    # 2. GMM Prior KL Divergence
    # Expand variational posterior parameters: (N, 1, D)
    mu_z_exp = mu_z.unsqueeze(1)
    var_z_exp = torch.exp(logvar_z).unsqueeze(1)
    logvar_z_exp = logvar_z.unsqueeze(1)

    # Prior parameters: (1, K, D)
    mu_c = model.mu_prior.unsqueeze(0)
    logvar_c = torch.clamp(model.logvar_prior.unsqueeze(0), min=-10.0, max=10.0)
    var_c = torch.exp(logvar_c)

    pi = F.softmax(model.pi_logits, dim=0).unsqueeze(0)  # (1, K)

    # Component-wise KL divergence:
    # 0.5 * sum_d [ logvar_c - logvar_z + (var_z + (mu_z - mu_c)^2) / var_c - 1 ]
    diff_sq = (mu_z_exp - mu_c) ** 2
    kl_components = 0.5 * torch.sum(
        logvar_c - logvar_z_exp + (var_z_exp + diff_sq) / (var_c + 1e-8) - 1.0,
        dim=-1,
    )  # (N, K)

    # Cluster entropy term: sum_c gamma_{ic} * log(gamma_{ic} / pi_c)
    entropy_term = torch.log(gamma + 1e-10) - torch.log(pi + 1e-10)  # (N, K)

    # Expectation over variational cluster responsibilities
    loss_gmm = torch.mean(torch.sum(gamma * (kl_components + entropy_term), dim=-1))

    return loss_recon, loss_gmm


def compute_graph_regularization_loss(
    z_cust: torch.Tensor,
    z_prod_proj: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    max_edges: int = 50000,
) -> torch.Tensor:
    """
    Computes bipartite graph topological regularization:
      L_graph = (1 / |E|) * sum_{(u, i) in E} W_{ui} * || z_u - proj(h_i) ||^2
    Smooths customer and product embeddings in latent space proportional to transaction weight.
    """
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return torch.tensor(0.0, device=z_cust.device)

    # Subsample edges if necessary to maintain O(1) memory during CPU training
    if num_edges > max_edges:
        perm = torch.randperm(num_edges)[:max_edges]
        u_idx = edge_index[0, perm]
        v_idx = edge_index[1, perm]
        weights = edge_weight[perm] if edge_weight is not None else torch.ones(len(perm), device=z_cust.device)
    else:
        u_idx = edge_index[0]
        v_idx = edge_index[1]
        weights = edge_weight if edge_weight is not None else torch.ones(num_edges, device=z_cust.device)

    z_u = z_cust[u_idx]
    z_v = z_prod_proj[v_idx]

    # Weighted Euclidean distance in latent space
    dist_sq = torch.sum((z_u - z_v) ** 2, dim=-1)
    weighted_dist = weights * dist_sq
    loss_graph = torch.mean(weighted_dist)

    return loss_graph


def compute_multitask_losses(
    y_hat_churn: torch.Tensor,
    y_hat_clv: torch.Tensor,
    y_churn: torch.Tensor,
    y_clv: torch.Tensor,
    y_hat_log_clv: torch.Tensor = None,
    pos_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Multi-Task Learning (MTL) losses:
      1. Binary Churn Classification: Binary Cross-Entropy (BCE) with class weighting
      2. Continuous CLV Regression: Mean Squared Logarithmic Error (MSLE)
    """
    churn_pred = y_hat_churn.squeeze(-1)
    churn_true = y_churn.float()

    # Binary Cross-Entropy with pos_weight
    # BCE = - [ w * y * log(p) + (1 - y) * log(1 - p) ]
    eps = 1e-7
    churn_pred_clipped = torch.clamp(churn_pred, eps, 1.0 - eps)
    bce_loss = - (
        pos_weight * churn_true * torch.log(churn_pred_clipped)
        + (1.0 - churn_true) * torch.log(1.0 - churn_pred_clipped)
    )
    loss_churn = torch.mean(bce_loss)

    # Continuous CLV MSLE: MSE(log1p(y_hat), log1p(y_true))
    if y_hat_log_clv is not None:
        log_pred = torch.clamp(y_hat_log_clv.squeeze(-1), min=0.0)
    else:
        log_pred = torch.log1p(torch.clamp(y_hat_clv.squeeze(-1), min=0.0))

    clv_true = torch.clamp(y_clv.float(), min=0.0)
    log_true = torch.log1p(clv_true)
    loss_clv = F.mse_loss(log_pred, log_true, reduction="mean")

    return loss_churn, loss_clv


def compute_hgvade_total_loss(
    model_output: dict[str, torch.Tensor],
    x_cust: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    y_churn: torch.Tensor,
    y_clv: torch.Tensor,
    model: HGVaDEModel,
    beta: float = 0.5,
    lambda_graph: float = 0.1,
    alpha_churn: float = 1.0,
    alpha_clv: float = 0.5,
    pos_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Computes all component losses and the unified multi-objective loss:
      L_total = L_recon + beta * L_gmm + lambda_graph * L_graph + alpha_churn * L_churn + alpha_clv * L_clv
    """
    loss_recon, loss_gmm = compute_elbo_and_gmm_loss(
        x_cust=x_cust,
        x_hat_cust=model_output["x_hat_cust"],
        mu_z=model_output["mu_z"],
        logvar_z=model_output["logvar_z"],
        gamma=model_output["gamma"],
        model=model,
    )

    loss_graph = compute_graph_regularization_loss(
        z_cust=model_output["z"],
        z_prod_proj=model_output["z_prod_proj"],
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    loss_churn, loss_clv = compute_multitask_losses(
        y_hat_churn=model_output["y_hat_churn"],
        y_hat_clv=model_output["y_hat_clv"],
        y_churn=y_churn,
        y_clv=y_clv,
        y_hat_log_clv=model_output.get("y_hat_log_clv"),
        pos_weight=pos_weight,
    )

    total_loss = (
        loss_recon
        + (beta * loss_gmm)
        + (lambda_graph * loss_graph)
        + (alpha_churn * loss_churn)
        + (alpha_clv * loss_clv)
    )

    return {
        "total_loss": total_loss,
        "loss_recon": loss_recon,
        "loss_gmm": loss_gmm,
        "loss_graph": loss_graph,
        "loss_churn": loss_churn,
        "loss_clv": loss_clv,
    }
