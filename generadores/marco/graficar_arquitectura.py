import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from generadores.marco.architecture import TimeVAE

trained = TimeVAE(latent_dim=8, dropout_rate=0.2, l2_reg=1e-4)
_ = trained(tf.zeros((1, 65, 3)))
trained.load_weights("vae_donors_weights.weights.h5")

enc, dec = trained.encoder, trained.decoder

# Reconstruimos el grafo en estilo Funcional, REUTILIZANDO las mismas capas
# (mismos pesos ya entrenados) -- solo para que plot_model pueda trazarlo.
inp = Input(shape=(65, 3), name="input_window")

x = enc.conv1(inp); x = enc.bn1(x); x = enc.act1(x); x = enc.drop1(x)
x = enc.conv2(x); x = enc.bn2(x); x = enc.act2(x); x = enc.drop2(x)
x = enc.flatten(x)
mu = enc.fc_mu(x)
logvar = enc.fc_logvar(x)

z = layers.Lambda(
    lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0])),
    name="reparameterize",
)([mu, logvar])

y = dec.fc(z); y = dec.bn_fc(y); y = dec.act_fc(y); y = dec.reshape(y)
y = dec.conv_t1(y); y = dec.bn_t1(y); y = dec.act_t1(y)
y = dec.conv_t2(y)
out = dec.cropping(y)

shadow_model = Model(inputs=inp, outputs=[out, mu, logvar], name="TimeVAE")

import os

# Ajusta esta ruta si instalaste Graphviz en otro disco o carpeta
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

tf.keras.utils.plot_model(
    shadow_model, to_file="arquitectura_vae.png",
    show_shapes=True, show_layer_names=True, dpi=150,
)