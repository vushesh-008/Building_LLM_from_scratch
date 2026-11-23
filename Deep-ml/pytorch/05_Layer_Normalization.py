import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    mean = torch.mean(X, dim=-1, keepdim=True)
    var = torch.var(X, dim = -1 , unbiased=False, keepdim=True)
    normalized = ( X - mean)/torch.sqrt(var + epsilon)

    result = gamma * normalized + beta 

    return result

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 8

    X = torch.randn(batch_size, seq_len, d_model)
    gamma = torch.ones(d_model)
    beta = torch.zeros(d_model)
    result = layer_normalization(X, gamma, beta)

    print("Input shape:", X.shape)
    print("Output shape:", result.shape)
    print("Mean:", torch.mean(result, dim=-1))
    print("Variance:", torch.var(result, dim=-1))
