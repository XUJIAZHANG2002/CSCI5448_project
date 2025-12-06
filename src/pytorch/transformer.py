
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model=64, d_ff=128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.seq(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=128):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab=1000, d_model=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


if __name__ == "__main__":
    model = MiniTransformer()
    x = torch.randint(0, 1000, (2, 10))  # (batch=2, seq=10)
    out = model(x)
    print("PyTorch output shape:", out.shape)
