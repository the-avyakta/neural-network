import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
   
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    attention_weights = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


x = torch.randn(1, 3, 4) 

output, weights = scaled_dot_product_attention(x, x, x)

print("Attention weights:\n", weights)
print("Output shape:", output.shape)

"""
Attention weights:
 tensor([[[0.7333, 0.0397, 0.2270],
         [0.2106, 0.5698, 0.2196],
         [0.4469, 0.0816, 0.4715]]])
Output shape: torch.Size([1, 3, 4])


"""
