import torch
import torch.nn.functional as F
from typing import Tuple


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X:             (seq_len, d_model)
        W_q, W_k, W_v: (d_model, d_model)

    Returns:
        Q, K, V: each of shape (seq_len, d_model)
    """
    Q = torch.matmul(X, W_q)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    K = torch.matmul(X, W_k)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    V = torch.matmul(X, W_v)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    return Q, K, V


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention (supports 2D single-head or 3D batched-heads).

    Args:
        Q: (..., seq_len, d_k)
        K: (..., seq_len, d_k)
        V: (..., seq_len, d_k)

    Returns:
        Attention output of shape (..., seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # (..., seq_len, d_k) @ (..., d_k, seq_len) -> (..., seq_len, seq_len)
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    # Numerically stable row-wise softmax -> (..., seq_len, seq_len)
    scaled_scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scaled_scores)
    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    # (..., seq_len, seq_len) @ (..., seq_len, d_k) -> (..., seq_len, d_k)
    return attention_weights @ V


def multi_head_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """
    Compute multi-head attention in parallel using .view() and .transpose().

    Args:
        Q, K, V: (seq_len, d_model)
        n_heads: Number of attention heads (must divide d_model evenly)

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads

    # 1. Split d_model -> (n_heads, d_k) and transpose to (n_heads, seq_len, d_k)
    Q_heads = Q.view(seq_len, n_heads, d_k).transpose(0, 1)
    K_heads = K.view(seq_len, n_heads, d_k).transpose(0, 1)
    V_heads = V.view(seq_len, n_heads, d_k).transpose(0, 1)

    # 2. Run self-attention across all heads in 1 batched operation -> (n_heads, seq_len, d_k)
    head_outputs = self_attention(Q_heads, K_heads, V_heads)

    # 3. Swap back to (seq_len, n_heads, d_k) and merge heads -> (seq_len, d_model)
    return head_outputs.transpose(0, 1).contiguous().view(seq_len, d_model)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters / Dimensions
    seq_len = 4   # Number of tokens
    d_model = 8   # Full embedding dimension
    n_heads = 2   # Number of heads -> d_k = d_model // n_heads = 4

    # Input token embeddings: (4, 8)
    X = torch.randn(seq_len, d_model)

    # Learnable projection weights: all (8, 8)
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)

    # Step 1: Project X into Q, K, V -> each (4, 8)
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print("Q, K, V shapes:", Q.shape, K.shape, V.shape)  # torch.Size([4, 8]) ...

    # Step 2: Reshape into 2 heads of (2, 4, 4), attend in parallel, and flatten back to (4, 8)
    output = multi_head_attention(Q, K, V, n_heads=n_heads)
    print("Output shape:", output.shape)  # torch.Size([4, 8])
