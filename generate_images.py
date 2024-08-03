import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the generator model
generator = load_model('generator_model.h5')

# Generate synthetic images
def generate_images(generator, num_images=10):
    noise = np.random.normal(0, 1, (num_images, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    for i in range(num_images):
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f"data/generated_image_{i}.png")
        plt.show()

# Generate and save synthetic images
generate_images(generator, num_images=10)
print("Image generation completed.")
