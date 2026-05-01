import torch
import torch.nn.functional as F
import numpy as np 

def custom_attention(Q, K, V):
    Q = Q.numpy()
    K = K.numpy()
    V = V.numpy()

    d_k = Q.shape[-1]
    score = (np.matmul(Q, K.transpose(0,2,1))/np.sqrt(d_k))
    
    exp = np.exp(score - np.max(score, keepdims=True, axis=-1))
    softmax = exp/np.sum(exp, keepdims=True, axis=-1)
    output = np.matmul(softmax, V)

    return output, softmax


def torch_attention(query, key, value):
   d_k = query.size(-1)
   score = (torch.matmul(query, key.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k, dtype=torch.float32)))
   weights = F.softmax(score, dim=-1)
   output = torch.matmul(weights, value)

   return output, weights


x = torch.randn(1, 3, 4) 

# output, weights = torch_attention(x, x, x)
# outputnp, weightsnp = custom_attention(x, x, x)

# print("Attention weights:\n", weights)
# print("Output shape:", output.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):  # d_model = size of each word vector & num_heads = number of parallel attentions
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # Each attention gets how much word vector 
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model) # “Take numbers X → mix them → output Y size, different values”
        self.k_linear = nn.Linear(d_model, d_model) # y = Wx + b
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #“Reshape the data, don’t change values
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # batches, words, features
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear layer
        out = self.out(out)
        
        return out


# print("Attention weights:\n", weightsnp)
# print("Output shape:", outputnp.shape)

"""
Attention weights:
 tensor([[[0.9242, 0.0408, 0.0350],
         [0.2170, 0.7689, 0.0141],
         [0.2922, 0.0221, 0.6857]]])
Output shape: torch.Size([1, 3, 4])


Attention weights:
 [[[0.9242068  0.04076316 0.03503005]
  [0.21702543 0.76886255 0.01411203]
  [0.29219195 0.02210925 0.6856988 ]]]
Output shape: (1, 3, 4)

"""
