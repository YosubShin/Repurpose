import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Clone the positional encoding to avoid in-place operations that cause DDP issues
        pe_slice = self.pe[:x.size(0), :].clone()
        x = x + pe_slice
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Register scale as a buffer to avoid DDP issues with shared parameters
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([self.d_k])))

    def forward(self, q, k, v, mask=None):
        # Scale should already be on the correct device as a buffer
        bs = q.size(0)

        # perform linear operation and split into num_heads
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k)

        # transpose to get dimensions bs * num_heads * seq_len * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add a dimension for heads
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)

        # apply attention
        x = torch.matmul(attention, v)

        # concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.0):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attention(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.0):
        super().__init__()

        # MultiHeadAttention for cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        # Norm and Feedforward layers remain similar
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        # Apply normalization
        x2 = self.norm_1(x)
        # Cross-attention, where context is used as key and value
        x = x + self.dropout_1(self.cross_attention(x2, context, context, mask))
        # Apply second normalization and feedforward network
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class CrossSelfEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.0):
        super().__init__()

        # MultiHeadAttention for self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # MultiHeadAttention for cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # Norm and Feedforward layers remain similar
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)  # Add an additional layer norm
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),  # You should have dropout here as well
            nn.Linear(d_ff, d_model)
        )

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        # Save the original input for residual connections
        orig_x = x

        # Normalize and perform self-attention
        x = self.norm_1(x)
        x2 = self.self_attention(x, x, x, mask)  # Self-attention uses x as query, key, value
        x = orig_x + self.dropout_1(x2)  # Apply residual connection and dropout

        # Normalize and perform cross-attention using the output of self-attention as query
        x = self.norm_2(x)
        x2 = self.cross_attention(x, context, context, mask)  # Cross-attention
        x = x + self.dropout_2(x2)  # Apply residual connection and dropout

        # Apply the final normalization and pass through feed-forward network
        x = self.norm_3(x)
        x2 = self.ff(x)
        x = x + self.dropout_2(x2)  # Final residual connection and dropout
        
        return x


class UniModalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff=2048):
        super(UniModalEncoder, self).__init__()
        self.mlp = MLP(input_dim, d_ff, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.mlp(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x