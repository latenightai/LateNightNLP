import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, x, w_key, w_query, w_value):
        super().__init__()
        self.x = x
        self.w_key = w_key
        self.w_query = w_query
        self.w_value = w_value

        x = torch.tensor(x, dtype=torch.float32)
        w_key = torch.tensor(w_key, dtype=torch.float32)
        w_query = torch.tensor(w_query, dtype=torch.float32)
        w_value = torch.tensor(w_value, dtype=torch.float32)
        
        keys = x @ w_key
        querys = x @ w_query
        values = x @ w_value

        # the attention score
        attn_scores = querys @ keys.T
        
        # with softmax
        attention_score_softmax = F.softmax(attn_scores, dim=-1)        
        attention_score_softmax = torch.tensor(attention_score_softmax)
    
        # softmax x corresponding values
        weighted_values = values[:,None] * attention_score_softmax.T[:,:,None]

        # sum of the weighted values
        outputs = weighted_values.sum(dim=0)
        print(outputs)

x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]

model = SelfAttention(x, w_key, w_query, w_value)


# this code has been used from the medium blog: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a