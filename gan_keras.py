import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, LeakyReLU, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
keras.backend.clear_session()
print(tf.__version__)

# Setup project tracking in wandb
import wandb
from wandb.keras import WandbCallback
wandb.init(project="GAN")
wandb.init(config={"hyper": "parameter"})

# Create Discriminator Model
discriminator = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    GlobalMaxPooling2D(),
    Dense(32, activation='relu'),
    Dense(1)
], name='discriminator')

# Create Generator Model
latent_dim = 128
generator = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    Dense(7*7*128),
    LeakyReLU(alpha=0.2),
    Reshape((7, 7, 128)),
    Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2D(1, (7, 7), padding='same', activation='sigmoid')
], name='generator')


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        # Sample random points in latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))

        # Convert random vector into fake image
        generated_imgs = self.generator(random_latent_vectors)

        # Add generated images to real images dataset
        imgs = tf.concat([generated_imgs, real_images], axis=0)

        # Create labels for generated and real images
        labels = tf.concat([tf.ones((batch_size, 1)),
                            tf.zeros((batch_size, 1))], axis=0)

        # Add noise to labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            preds = self.discriminator(imgs)
            d_loss = self.loss_fn(labels, preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train generator (dont update weights of discriminator)
        # Sample random points in latent space
        random_latent_vecotrs = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Create misleading labels
        mislead_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(random_latent_vecotrs))
            g_loss = self.loss_fn(mislead_labels, preds)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {'disc_loss': d_loss, 'gener_loss': g_loss}


# Prepare dataset
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype('float32')/255.
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

#  Instantiate the model
gan = GAN(discriminator=discriminator,
          generator=generator,
          latent_dim=latent_dim)

gan.compile(d_optimizer=keras.optimizers.SGD(learning_rate=0.0003),
            g_optimizer=keras.optimizers.SGD(learning_rate=0.0003),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))

# Train the model
gan.fit(dataset, epochs=100, callbacks=[WandbCallback()])
