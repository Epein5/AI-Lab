import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class ImputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(d_model, vocab_size)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int ,seq_len:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float  = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/torch.sqrt(std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # x = self.linear_1(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.linear_2(x)
        # return x
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentation(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attentaion(query, key , value, mask , dropout: nn.Dropout):
        d_k = query.shape[-1]

        attentation_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attentation_score.masked_fill(mask == 0, -1e9)
        attentation_score = attentation_score.softmax(dim=-1)

        if dropout is not None:
            attentation_score = dropout(attentation_score)
        
        attentaion  = attentation_score @ value

        return (attentaion), attentation_score


    def forward(self,q, k, v , mask ):

        query = self.w_q(q) 
        key = self.w_k(k) 
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attentaion_scores = MultiHeadAttentation.attentaion(query, key, value, mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1 , self.h * self.d_k)

        return self.w_o(x)
    

class ResudialConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def fowward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attentaion_block: MultiHeadAttentation, feed_forward_block: FeedForward, dropout:float ):
        super().__init__()
        self.self_attentaion_block = self_attentaion_block
        self. feed_forward_block = feed_forward_block
        self.res_connection = nn.ModuleList([ResudialConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.res_connection[0](x,self.self_attentaion_block(x, x, x, src_mask))
        x = self.res_connection[1](x, self.feed_forward_block(x))
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forwaed(self, x , mask):
        for layer in self.layers:
            x = layer(x,layer)
        return self.norm(x)
