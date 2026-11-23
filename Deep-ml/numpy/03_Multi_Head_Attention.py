import numpy as np
from typing import Tuple


def compute_qkv(
    X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X:           (seq_len, d_model)
        W_q, W_k, W_v: (d_model, d_model)

    Returns:
        Q, K, V: each of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    K = np.dot(X, W_k)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    V = np.dot(X, W_v)  # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    return Q, K, V


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention for a single head.

    Args:
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Numerically stable row-wise softmax -> (seq_len, seq_len)
    scaled_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scaled_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # (seq_len, seq_len) @ (seq_len, d_k) -> (seq_len, d_k)
    return attention_weights @ V


def multi_head_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: (seq_len, d_model)
        n_heads: Number of attention heads (must divide d_model evenly)

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # 1. Split (seq_len, d_model) -> list of n_heads arrays, each (seq_len, d_k)
    Q_heads = np.split(Q, n_heads, axis=-1)
    K_heads = np.split(K, n_heads, axis=-1)
    V_heads = np.split(V, n_heads, axis=-1)

    # 2. Run self-attention on each head -> list of n_heads arrays, each (seq_len, d_k)
    head_outputs = [
        self_attention(q, k, v) for q, k, v in zip(Q_heads, K_heads, V_heads)
    ]

    # 3. Glue heads back together -> (seq_len, d_model)
    return np.concatenate(head_outputs, axis=-1)


if __name__ == "__main__":
    np.random.seed(42)

    # Hyperparameters / Dimensions
    seq_len = 4   # Number of tokens
    d_model = 8   # Full embedding dimension
    n_heads = 2   # Number of heads -> d_k = d_model // n_heads = 4

    # Input token embeddings: (4, 8)
    X = np.random.randn(seq_len, d_model)

    # Learnable projection weights: all (8, 8)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)

    # Step 1: Project X into Q, K, V -> each (4, 8)
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print("Q, K, V shapes:", Q.shape, K.shape, V.shape)  # (4, 8) (4, 8) (4, 8)

    # Step 2: Split into 2 heads of (4, 4), attend, and concatenate back to (4, 8)
    output = multi_head_attention(Q, K, V, n_heads=n_heads)
    print("Output shape:", output.shape)  # (4, 8)