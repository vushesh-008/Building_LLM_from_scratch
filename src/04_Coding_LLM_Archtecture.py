import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])

        self.trf_blocks = nn.Sequential(
            *[nn.DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_length = in_idx.shape
        tok_embeds = self.token_emb(in_idx) # (batch_size, seq_length, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_length, device=in_idx.device)) # (seq_length, emb_dim)
        x = tok_embeds + pos_embeds # (batch_size, seq_length, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x) # (batch_size, seq_length, emb_dim)
        x = self.final_norm(x) # (batch_size, seq_length, emb_dim)
        logits = self.out_head(x) # (batch_size, seq_length, vocab_size)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1 , keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.scale * x_norm + self.shift
        return out
    
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * ( 1 + torch.tanh( torch.sqrt(torch.sqrt( torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow( x, 3)))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
