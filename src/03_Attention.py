import torch
from torch import nn
class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_key = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_value = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # Dropout layer with specified dropout rate
        self.register_buffer("mask", 
                             torch.triu(torch.ones(context_length, context_length), 
                                        diagonal=1).bool()) # Registering the upper triangular mask as a buffer
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values = self.W_value(x) # (b, num_tokens, d_out)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Apply the upper triangular mask to the attention scores
        attn_weights = torch.softmax(attn_scores, dim=-1) # Normalize the attention scores to get attention weights
        attn_weights = self.dropout(attn_weights) # Apply dropout to the attention weights

        output = attn_weights @ values # Compute the output as a weighted sum of the values
        return output
    


torch.manual_seed(123) # Set random seed for reproducibility
context_length = batch.shape[1]

ca = CasualAttention(d_in, d_out, context_length=context_length, dropout=0.0)
context_vecs = ca(batch)
print("Context vectors shape: ", context_vecs.shape)