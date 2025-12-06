
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, d_ff=128):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn = self.mha(x, x)
        x = self.norm1(x + attn)
        f = self.ffn(x)
        return self.norm2(x + f)


class MiniTransformer(Model):
    def __init__(self, vocab=1000, d_model=64, num_layers=2):
        super().__init__()
        self.embed = layers.Embedding(vocab, d_model)
        self.layers_t = [TransformerBlock(d_model) for _ in range(num_layers)]
        self.out_proj = layers.Dense(vocab)

    def call(self, ids):
        x = self.embed(ids)
        for layer in self.layers_t:
            x = layer(x)
        return self.out_proj(x)


if __name__ == "__main__":
    model = MiniTransformer()
    x = tf.random.uniform((2, 10), maxval=999, dtype=tf.int32)
    out = model(x)
    print("TF output shape:", out.shape)
