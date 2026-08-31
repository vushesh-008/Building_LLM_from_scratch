import torch
from torch import nn
torch.manual_seed(123) # Set random seed for reproducibility
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

        b , num_tokens , d_in = x.shape

        keys = self.W_key(x) # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values = self.W_value(x) # (b, num_tokens, d_out)

        attn_scores = queries @ keys.transpose(-2, -1) # Compute attention scores by multiplying queries with the transpose of keys
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Apply the upper triangular mask to the attention scores
        attn_weights = torch.softmax(attn_scores/ (keys.shape[-1] ** 0.5), dim=-1)

        attn_weights = self.dropout(attn_weights) # Apply dropout to the attention weights

        context_vectors = attn_weights @ values # Compute the output as a weighted sum of the values
        return context_vectors


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()
        self.heads = nn.ModuleList([
            CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads] # Get the output from each attention head
        return torch.cat(head_outputs, dim=-1) # Concatenate the outputs from all heads along the last dimension


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        # Ensure that the output dimension is divisible by the number of heads
        # This is necessary because we will split the output dimension into equal parts for each head
        # we don't just concatanate the outputs from each head, we also need to ensure that the output dimension of each head is the same
        assert (d_out % num_heads ==0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Calculate the dimension of each head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # Linear layer to project input to query space
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # Linear layer to project input to key space
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # Linear layer to project input to value space

        # Linear layer to project the concatenated output of all heads back to the original output dimension
        self.out_proj = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout) # Dropout layer with specified dropout rate
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()) # Registering the upper triangular mask as a buffer

    def forward(self, x):
        b, num_tokens, d_in = x.shape # Get the batch size, number of tokens, and input dimension from the shape of the input tensor

        keys = self.W_key(x) # Tensor shape : (b, num_tokens, d_out) - Project the input to key space using the linear layer
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape the keys, queries, and values to separate the heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # Transpose the keys to have shape (b, num_heads, num_tokens, head_dim) - This is necessary for the attention calculation
        queries = queries.transpose(1, 2) # queries shape: [b, h, T, d_k]
        values = values.transpose(1, 2) # values shape: [b, h, T, d_k]

        # [b, h, T, d_k] @ [b, h, d_k, T] -> fails since d_k != T
        # [b, h, T, d_k] @ [b, h, T, d_k] -> Works since d_k == d_k
        attn_scores = queries @ keys.transpose(2, 3) # Compute attention scores by multiplying queries with the transpose of keys
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Apply the upper triangular mask to the attention scores

        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1) # Apply softmax to the attention scores to get attention weights
        attn_weights = self.dropout(attn_weights) # Apply dropout to the attention weights

        # attn_weights : [b, h, T, T] 
        # values : [b, h, T, d_k]
        # matrix muitplication : [b, h, T, T] @ [b, h, T, d_k] -> [b, h, T, d_k]
        # Transpose : [b, h, T, d_k] -> [b, T, h, d_k] -> [b, T, h * d_k] = [b, T, d_out]
        context_vectors = (attn_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out) # Reshape the context vectors to have shape (b, num_tokens, d_out) - This is necessary to concatenate the outputs from all heads

        output = self.out_proj(context_vectors) # Project the concatenated output of all heads back to the original output dimension using the linear layer

        return output           


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in , d_out = 3, 2 # Set the input and output dimensions to the embedding dimension of the tokens
print("Shape for input dimesions and output dimensions: ", d_in, d_out)

batch = torch.stack((inputs, inputs), dim=0)
# 2 inputs with 6 tokens each, and each token has embedding dimension 3
print("Batch shape: ", batch.shape)
print("Batch: \n", batch)


context_length = batch.shape[1]

# ca = CasualAttention(d_in, d_out, context_length=context_length, dropout=0.0)
# context_vecs = ca(batch)
# print("Context vectors shape: ", context_vecs.shape)
# print("Context vectors: \n", context_vecs)

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

d_out = 1
mha2 = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha2(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)