import torch
from typing import Tuple


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices from input X and weight matrices.

    Args:
        X:   (seq_len, d_model)
        W_q: (d_model, d_k)
        W_k: (d_model, d_k)
        W_v: (d_model, d_v)

    Returns:
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_v)
    """
    Q = torch.matmul(X, W_q)  # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
    K = torch.matmul(X, W_k)  # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
    V = torch.matmul(X, W_v)  # (seq_len, d_model) @ (d_model, d_v) -> (seq_len, d_v)
    return Q, K, V


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape   (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    # Numerically stable row-wise softmax -> (seq_len, seq_len)
    scores_shifted = scores - torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores_shifted)
    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
    return attention_weights @ V


if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters / Dimensions
    seq_len = 4   # Number of tokens in the sequence
    d_model = 8   # Input embedding dimension (e.g., 768 in BERT)
    d_k = 4       # Query / Key projection dimension
    d_v = 6       # Value projection dimension (often equal to d_k, but doesn't have to be)

    # Input token embeddings: (4, 8)
    X = torch.randn(seq_len, d_model)

    # Learnable projection weights
    W_q = torch.randn(d_model, d_k)  # (8, 4)
    W_k = torch.randn(d_model, d_k)  # (8, 4)
    W_v = torch.randn(d_model, d_v)  # (8, 6)

    # Step 1: Project X into Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print("Q shape:", Q.shape)  # torch.Size([4, 4])
    print("K shape:", K.shape)  # torch.Size([4, 4])
    print("V shape:", V.shape)  # torch.Size([4, 6])

    # Step 2: Run single-head self-attention
    output = self_attention(Q, K, V)
    print("Output shape:", output.shape)  # torch.Size([4, 6])
