import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    x1 , x2 = x.chunk(2, dim = -1)
    swish = x2 / (1 + torch.exp(-x2))

    scores = x1 * swish

    return scores


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    result = SwiGLU(x)
    print("Input shape:", x.shape)
    print("Output shape:", result.shape)