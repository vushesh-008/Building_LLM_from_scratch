import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
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
    Q = np.dot(X, W_q)  # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
    K = np.dot(X, W_k)  # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
    V = np.dot(X, W_v)  # (seq_len, d_model) @ (d_model, d_v) -> (seq_len, d_v)
    return Q, K, V


def self_attention(Q, K, V):
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
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Numerically stable row-wise softmax -> (seq_len, seq_len)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
    return attention_weights @ V


if __name__ == "__main__":
    np.random.seed(42)

    # Hyperparameters / Dimensions
    seq_len = 4   # Number of tokens in the sequence
    d_model = 8   # Input embedding dimension (e.g., 768 in BERT)
    d_k = 4       # Query / Key projection dimension
    d_v = 6       # Value projection dimension (often equal to d_k, but doesn't have to be)

    # Input token embeddings: (4, 8)
    X = np.random.randn(seq_len, d_model)

    # Learnable projection weights
    W_q = np.random.randn(d_model, d_k)  # (8, 4)
    W_k = np.random.randn(d_model, d_k)  # (8, 4)
    W_v = np.random.randn(d_model, d_v)  # (8, 6)

    # Step 1: Project X into Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print("Q shape:", Q.shape)  # (4, 4)
    print("K shape:", K.shape)  # (4, 4)
    print("V shape:", V.shape)  # (4, 6)

    # Step 2: Run single-head self-attention
    output = self_attention(Q, K, V)
    print("Output shape:", output.shape)  # (4, 6)  