import numpy as np


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.

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


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention (e.g., causal attention in a decoder).

    Args:
        Q:    (seq_len, d_k)
        K:    (seq_len, d_k)
        V:    (seq_len, d_v)
        mask: (seq_len, seq_len) with 0 at allowed positions and -inf at blocked positions

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_length, d_k = Q.shape

    # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Apply mask before softmax: blocked positions become -inf -> (seq_len, seq_len)
    intermediate_scores = scores + mask

    # Numerically stable row-wise softmax -> (seq_len, seq_len)
    scaled_scores = intermediate_scores - np.max(intermediate_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scaled_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
    return attention_weights @ V


if __name__ == "__main__":
    np.random.seed(42)

    # Hyperparameters / Dimensions
    seq_len = 4   # Number of tokens
    d_model = 8   # Input embedding dimension
    d_k = 4       # Query / Key dimension
    d_v = 4       # Value dimension

    # Input token embeddings: (4, 8)
    X = np.random.randn(seq_len, d_model)

    # Learnable projection weights
    W_q = np.random.randn(d_model, d_k)  # (8, 4)
    W_k = np.random.randn(d_model, d_k)  # (8, 4)
    W_v = np.random.randn(d_model, d_v)  # (8, 4)

    # Causal mask: 0 on and below the diagonal, -inf strictly above the diagonal -> (4, 4)
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

    # Step 1: Project X into Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print("Q shape:", Q.shape)        # (4, 4)
    print("K shape:", K.shape)        # (4, 4)
    print("V shape:", V.shape)        # (4, 4)
    print("Mask shape:", mask.shape)  # (4, 4)

    # Step 2: Run masked self-attention
    output = masked_attention(Q, K, V, mask)
    print("Output shape:", output.shape)  # (4, 4)