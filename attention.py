import torch
import torch.nn.functional as F
import numpy as np 


def torch_attention(query, key, value):
   d_k = query.size(-1)
   score = (torch.matmul(query, key.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k, dtype=torch.float32)))
   weights = F.softmax(score, dim=-1)
   output = torch.matmul(weights, value)

   return output, weights


x = torch.randn(1, 3, 4) 

output, weights = torch_attention(x, x, x)


print("Attention weights:\n", weights)
print("Output shape:", output.shape)


"""
Attention weights:
 tensor([[[0.9242, 0.0408, 0.0350],
         [0.2170, 0.7689, 0.0141],
         [0.2922, 0.0221, 0.6857]]])
Output shape: torch.Size([1, 3, 4])


"""
