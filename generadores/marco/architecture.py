import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class TimeVAEEncoder(layers.Layer):
    def __init__(self, latent_dim=8, dropout_rate=0.2, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        reg = regularizers.l2(l2_reg)
        
        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same", kernel_regularizer=reg)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        self.drop1 = layers.Dropout(dropout_rate)
        
        self.conv2 = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding="same", kernel_regularizer=reg)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation("relu")
        self.drop2 = layers.Dropout(dropout_rate)
        
        self.flatten = layers.Flatten()
        self.fc_mu = layers.Dense(latent_dim, kernel_regularizer=reg)
        self.fc_logvar = layers.Dense(latent_dim, kernel_regularizer=reg)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

class TimeVAEDecoder(layers.Layer):
    def __init__(self, output_length=65, num_channels_out=3, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        reg = regularizers.l2(l2_reg)
        
        self.fc = layers.Dense(17 * 64, kernel_regularizer=reg)
        self.bn_fc = layers.BatchNormalization()
        self.act_fc = layers.Activation("relu")
        self.reshape = layers.Reshape((17, 64))
        
        self.conv_t1 = layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding="same", kernel_regularizer=reg)
        self.bn_t1 = layers.BatchNormalization()
        self.act_t1 = layers.Activation("relu")
        
        self.conv_t2 = layers.Conv1DTranspose(filters=num_channels_out, kernel_size=3, strides=2, padding="same")
        self.cropping = layers.Cropping1D(cropping=(0, 3))

    def call(self, inputs, training=False):
        x = self.fc(inputs)
        x = self.bn_fc(x, training=training)
        x = self.act_fc(x)
        x = self.reshape(x)
        
        x = self.conv_t1(x)
        x = self.bn_t1(x, training=training)
        x = self.act_t1(x)
        
        x = self.conv_t2(x)
        return self.cropping(x)

class TimeVAE(Model):
    def __init__(self, latent_dim=8, dropout_rate=0.2, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.encoder = TimeVAEEncoder(latent_dim=latent_dim, dropout_rate=dropout_rate, l2_reg=l2_reg)
        self.decoder = TimeVAEDecoder(output_length=65, num_channels_out=3, l2_reg=l2_reg)

    def call(self, inputs, training=False):
        mu, logvar = self.encoder(inputs, training=training)
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(0.5 * logvar) * eps
        x_hat = self.decoder(z, training=training)
        return x_hat, mu, logvar