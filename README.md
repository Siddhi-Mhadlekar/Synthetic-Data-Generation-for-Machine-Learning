# synthetic_data_generation
# Synthetic Data Generation for Machine Learning

## Objective

Develop a system to generate synthetic images of cracked phone screens using Generative Adversarial Networks (GANs). This project aims to create realistic images that can be used for training computer vision models, data augmentation, and other machine learning applications.

## Overview

This project leverages a cracked phone image dataset to train a Generative Adversarial Network (GAN). The GAN will be trained to generate synthetic images of cracked phone screens, which can be used to improve the performance of machine learning models in recognizing damage patterns on phone screens. The project uses PySpark for data preprocessing and TensorFlow for GAN implementation.

## Project Goals

- **Data Preprocessing**: Load and preprocess the cracked phone images for training.
- **GAN Training**: Define and train a GAN to generate synthetic cracked phone images.
- **Image Generation**: Use the trained GAN to produce new synthetic images of cracked phone screens.
- **Evaluation**: Evaluate the quality of the generated images and their usefulness for downstream tasks.

## Services Used

- **PySpark**: For preprocessing and managing large datasets.
- **TensorFlow**: For implementing and training the Generative Adversarial Network (GAN).
- **NumPy**: For numerical operations and data manipulation.
- **Pandas**: For handling and processing data.
- **Matplotlib**: For visualizing the generated images.

## Dataset

- **Cracked Phone Images Dataset**: The dataset used in this project contains images of cracked phone screens. You can download and use the dataset from  [Cracked Phone Dataset](https://www.kaggle.com/datasets/dataclusterlabs/mobile-phone-image-dataset). Make sure to place the dataset in the `data/images` directory.

## How to Execute

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/synthetic_data_generation.git
   cd synthetic_data_generation

2. Install dependencies: pip install -r requirements.txt
3. Run the scripts:
   ```bash
    python data_preprocessing.py
    python train_gan.py
    python generate_images.py
