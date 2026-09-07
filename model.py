import torch
import torch.nn as nn
from torch.nn import functional as F

N_CASES = 64
N_PIECES = 13
N_META = 6


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, head_size, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout
        self.head_size_sr = head_size ** 0.5
    
    def forward(self, x, biais=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_heads, T, head_size)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=biais.to(q.dtype) if biais is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1/self.head_size_sr # redondant
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)



class FFN(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, dropout):
        super().__init__()
        self.w1 = nn.Linear(in_f, n_hidden)
        self.w2 =  nn.Linear(n_hidden, out_f)
        self.v = nn.Linear(in_f, n_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ((Swish_1(x @ W1)) * (x @ V)) @ W2
        x_w1 = self.w1(x)
        Swish_1 = x_w1 * torch.sigmoid(x_w1)
        x_v = self.v(x)
        out = self.w2(Swish_1 * x_v)
        return self.dropout(out)



class EncoderBlock(nn.Module): # no mask -> decoder become encoder
    def __init__(self, emb_dim, n_heads, dropout):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h

        self.multihead = MultiHeadAttention(n_heads, emb_dim, head_size, dropout)
        self.FFN = FFN(emb_dim, int(emb_dim*(8/3)), emb_dim, dropout) # 8/3 due to swiglu TODO: find a better /2 than raw 8/3
        self.ln1 = torch.nn.RMSNorm(emb_dim)
        self.ln2 = torch.nn.RMSNorm(emb_dim)

    
    def forward(self, x, biais=None):
        x = x + self.multihead(self.ln1(x), biais) # (B, T, C)
        x = x + self.FFN(self.ln2(x)) # (B, T, C)
        return x


        

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, context_len, n_block, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_position = nn.Embedding(context_len, emb_dim)
        self.emb_meta = nn.Linear(N_META, emb_dim)

        self.blocks = nn.ModuleList(
            [EncoderBlock(emb_dim, n_heads, dropout) for _ in range(n_block)]
        )

        self.biais_attention = nn.Parameter(torch.zeros(1, n_heads, context_len, context_len))
        self.ln = nn.RMSNorm(emb_dim)

        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU(), 
            nn.Linear(emb_dim, 1)
        )

        self.pol_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.pol_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.pol_scale = emb_dim ** 0.5

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / ((2 * n_block)**0.5))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, squares, meta, legal_mask=None):

        B, n_cases = squares.shape # (B, 64)

        x = squares

        tok_emb = self.emb(x) # (B, 64, C)
        pos_emb = self.emb_position(torch.arange(n_cases, device=squares.device)) # (B, 64, C)
        meta_emb = self.emb_meta(meta).unsqueeze(1) # (B, 1, C) -> chaque case reçoit les infos meta

        x = tok_emb + pos_emb + meta_emb # (B, 64, C)

        for bloc in self.blocks:
            x = bloc(x, self.biais_attention) # (B, 64, C)
        x = self.ln(x)

        value = torch.tanh(self.value_head(x.mean(dim=1))) # (B, 1)

        Qp = self.pol_q(x) # (B, 64, C)
        Kp = self.pol_k(x) # (B, 64, C)
        policy = (Qp @ Kp.transpose(-2, -1)) / self.pol_scale # (B, 64, 64)
 
        if legal_mask is not None:
            policy = policy.masked_fill(~legal_mask, torch.finfo(policy.dtype).min) # opérateur not niveau bit
 
        policy = policy.flatten(1) # (B, 64*64)

        return policy, value
