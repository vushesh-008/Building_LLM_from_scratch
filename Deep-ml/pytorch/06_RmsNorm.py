import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMSNorm to the input tensor.

    Parameters:
        x   : torch.Tensor of shape (batch_size, features)
        g   : torch.Tensor of shape (features,) - gain parameter
        eps : float - small constant for numerical stability

    Returns:
        torch.Tensor of same shape as x
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

    return (x/rms) * g


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 8
    x = torch.randn(batch_size, seq_len, d_model)
    g = torch.ones(d_model)
    result = rmsnorm(x, g)
    print("Input shape:", x.shape)
    print("Output shape:", result.shape)
    print("Mean:", torch.mean(result, dim=-1))
    print("Variance:", torch.var(result, dim=-1))