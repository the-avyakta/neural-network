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

output, weights = torch_attention(x, x, x)
outputnp, weightsnp = custom_attention(x, x, x)

print("Attention weights:\n", weights)
print("Output shape:", output.shape)


print("Attention weights:\n", weightsnp)
print("Output shape:", outputnp.shape)

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
