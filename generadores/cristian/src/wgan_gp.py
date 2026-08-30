from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .data import ChannelNormalizer, make_tf_dataset
from .io import RunMetadata, save_checkpoint, save_json, save_loss_history, save_run_metadata
from .models import build_critic, build_generator


@dataclass
class TrainConfig:
    latent_dim: int = 100
    batch_size: int = 64
    epochs: int = 5000
    n_critic: int = 5
    lambda_gp: float = 10.0
    learning_rate: float = 1e-4
    beta_1: float = 0.0
    beta_2: float = 0.9
    sample_interval: int = 200
    seed: int = 42


class WGAN_GP:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.generator = build_generator(config.latent_dim)
        self.critic = build_critic()
        self.g_optimizer = Adam(
            learning_rate=config.learning_rate,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
        )
        self.c_optimizer = Adam(
            learning_rate=config.learning_rate,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
        )

    @staticmethod
    def wasserstein_loss(real_scores: tf.Tensor, fake_scores: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

    def gradient_penalty(
        self,
        real_windows: tf.Tensor,
        fake_windows: tf.Tensor,
    ) -> tf.Tensor:
        batch_size = tf.shape(real_windows)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_windows + (1.0 - alpha) * fake_windows
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def train_critic_step(self, real_windows: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(real_windows)[0]
        latent_dim = self.config.latent_dim
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as tape:
            fake_windows = self.generator(noise, training=True)
            real_scores = self.critic(real_windows, training=True)
            fake_scores = self.critic(fake_windows, training=True)
            gp = self.gradient_penalty(real_windows, fake_windows)
            c_loss = self.wasserstein_loss(real_scores, fake_scores) + self.config.lambda_gp * gp

        gradients = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return c_loss, tf.reduce_mean(real_scores), tf.reduce_mean(fake_scores)

    @tf.function
    def train_generator_step(self, batch_size: tf.Tensor) -> tf.Tensor:
        latent_dim = self.config.latent_dim
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as tape:
            fake_windows = self.generator(noise, training=True)
            fake_scores = self.critic(fake_windows, training=True)
            g_loss = -tf.reduce_mean(fake_scores)
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return g_loss

    def train(
        self,
        train_windows: np.ndarray,
        *,
        run_dir: Path,
        normalizer: ChannelNormalizer,
    ) -> Model:
        tf.keras.utils.set_random_seed(self.config.seed)
        run_dir.mkdir(parents=True, exist_ok=True)

        normalized = normalizer.normalize(train_windows).astype(np.float32)
        dataset = make_tf_dataset(
            normalized,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )

        metadata = RunMetadata.now(
            seed=self.config.seed,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            latent_dim=self.config.latent_dim,
            n_critic=self.config.n_critic,
            lambda_gp=self.config.lambda_gp,
            learning_rate=self.config.learning_rate,
            n_train_windows=len(train_windows),
        )
        save_run_metadata(metadata, run_dir / "run_metadata.json")
        save_json(normalizer.to_dict(), run_dir / "normalizer.json")

        history: list[dict] = []
        batch_size_tensor = tf.constant(self.config.batch_size, dtype=tf.int32)

        for epoch in range(self.config.epochs):
            c_loss_epoch = []
            g_loss_epoch = []
            real_score_epoch = []
            fake_score_epoch = []

            for real_batch in dataset:
                for _ in range(self.config.n_critic):
                    c_loss, real_score, fake_score = self.train_critic_step(real_batch)
                    c_loss_epoch.append(float(c_loss.numpy()))
                    real_score_epoch.append(float(real_score.numpy()))
                    fake_score_epoch.append(float(fake_score.numpy()))

                g_loss = self.train_generator_step(batch_size_tensor)
                g_loss_epoch.append(float(g_loss.numpy()))

            record = {
                "epoch": epoch,
                "c_loss": float(np.mean(c_loss_epoch)),
                "g_loss": float(np.mean(g_loss_epoch)),
                "c_real_score": float(np.mean(real_score_epoch)),
                "c_fake_score": float(np.mean(fake_score_epoch)),
            }
            history.append(record)

            if epoch % self.config.sample_interval == 0 or epoch == self.config.epochs - 1:
                print(
                    f"epoch {epoch:5d} | c_loss={record['c_loss']:.4f} "
                    f"g_loss={record['g_loss']:.4f} "
                    f"D(real)={record['c_real_score']:.4f} D(fake)={record['c_fake_score']:.4f}"
                )
                checkpoint = save_checkpoint(
                    self.generator,
                    self.critic,
                    run_dir / "checkpoints",
                    epoch=epoch,
                )
                metadata.checkpoint = str(checkpoint)

        save_loss_history(history, run_dir / "loss_history.csv")
        save_run_metadata(metadata, run_dir / "run_metadata.json")
        return self.generator

    def generate(
        self,
        n_samples: int,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        noise = tf.random.normal([n_samples, self.config.latent_dim])
        generated = self.generator(noise, training=False)
        return generated.numpy()
