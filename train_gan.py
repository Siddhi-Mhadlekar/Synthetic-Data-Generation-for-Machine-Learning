import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import pyarrow.parquet as pq

# Load and preprocess data from Parquet file
def load_data(parquet_file):
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    # Assuming image data is stored in an appropriate format, e.g., lists or arrays
    images = np.array([np.array(img) for img in df['image_data']])
    images = (images.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
    return images

X_train = load_data('data/processed/images.parquet')
X_train = np.expand_dims(X_train, axis=3)  # Ensure the correct shape

# Define the GAN components
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=100),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator()

gan = tf.keras.Sequential([generator, discriminator])
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
def train_gan(gan, generator, discriminator, epochs=10000, batch_size=128):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

train_gan(gan, generator, discriminator)
generator.save('generator_model.h5')
print("GAN training completed.")
