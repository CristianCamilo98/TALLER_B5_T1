import tensorflow as tf

def reconstruction_loss(x, x_hat):
    """Pérdida Huber promediada por batch y dimensiones (igual que reduction='mean' de PyTorch)"""
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return huber(x, x_hat)

def kl_divergence_free_bits(mu, logvar, free_bits=0.25):
    kl_per_dim = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
    kl_per_dim_floored = tf.maximum(kl_per_dim, free_bits)
    kl_sum = tf.reduce_sum(kl_per_dim_floored, axis=-1)
    return tf.reduce_mean(kl_sum)