import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    d = x.shape[1]
    freqs = torch.arange(0,d,2, dtype = torch.float32)
    theta = 1/ base ** (freqs/d)

    angles = torch.outer(positions.float(), theta)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[: , 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x, dtype=torch.float32)
    out[:, 0::2] = out_even
    out[:, 1::2] = out_odd
    
    return out

if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 8
    d_model = 4
    x = torch.randn(seq_len, d_model)
    positions = torch.arange(seq_len)
    result = apply_rope(x, positions)
    print("Input shape:", x.shape)
    print("Output shape:", result.shape)