import tensorflow as tf

# Sample data
x = tf.random.normal((1000, 32))
y = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

# Build data pipeline declaratively
dataset = (
    tf.data.Dataset.from_tensor_slices((x, y))
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# Iterate
for batch_x, batch_y in dataset:
    # framework handles loading and batching internally
    print(batch_x.shape, batch_y.shape)
