import os
import random
import numpy as np
import tensorflow as tf
import yaml
import logging
from datetime import datetime

# Set random seed for reproducibility
def set_random_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and TensorFlow for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to: {seed}")

# Setup logger for the project
def setup_logger(log_file='train.log', log_level=logging.INFO):
    """
    Sets up a logger for the project to track training and evaluation progress.
    
    Args:
        log_file (str): Path to log file (default is 'train.log').
        log_level (int): Logging level (default is INFO).
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create a stream handler to output logs to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    
    # Create log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Load configuration from YAML file
def load_config(config_path='configs/config.yaml'):
    """
    Loads configuration settings from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    logging.info(f"Loaded config from {config_path}")
    return config

# Save model checkpoints
def save_model_checkpoint(model, checkpoint_dir, epoch, model_name='uigo_model'):
    """
    Saves the model checkpoint during training.
    
    Args:
        model (tf.keras.Model): The model to save.
        checkpoint_dir (str): Directory to save the checkpoint.
        epoch (int): The current training epoch.
        model_name (str): The model name (default is 'uigo_model').
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.h5")
    model.save(checkpoint_path)
    logging.info(f"Saved model checkpoint to {checkpoint_path}")

# Load model from checkpoint
def load_model_from_checkpoint(model, checkpoint_path):
    """
    Loads a model from a checkpoint.
    
    Args:
        model (tf.keras.Model): The model to load weights into.
        checkpoint_path (str): Path to the model checkpoint file.
    
    Returns:
        model (tf.keras.Model): Model with loaded weights.
    """
    model.load_weights(checkpoint_path)
    logging.info(f"Loaded model from {checkpoint_path}")
    return model

# Data augmentation (example)
def data_augmentation(image, mask):
    """
    Example of simple data augmentation (flip and rotation).
    
    Args:
        image (tf.Tensor): Input image tensor.
        mask (tf.Tensor): Corresponding mask tensor.
    
    Returns:
        image, mask (tf.Tensor, tf.Tensor): Augmented image and mask.
    """
    # Random flip
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    
    image = tf.image.random_flip_up_down(image)
    mask = tf.image.random_flip_up_down(mask)
    
    # Random rotation
    image = tf.image.rot90(image, k=random.randint(0, 3))
    mask = tf.image.rot90(mask, k=random.randint(0, 3))
    
    return image, mask

# Normalize images
def normalize_image(image):
    """
    Normalizes an image to the range [0, 1].
    
    Args:
        image (tf.Tensor): Input image tensor.
    
    Returns:
        image (tf.Tensor): Normalized image tensor.
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # Normalizing to [0, 1]
    return image

# Create directory if not exists
def create_directory(directory):
    """
    Creates a directory if it doesn't already exist.
    
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")

# Function to log the model parameters
def log_model_summary(model, log_file='model_summary.log'):
    """
    Logs the model summary to a file.
    
    Args:
        model (tf.keras.Model): The model whose summary needs to be logged.
        log_file (str): Path to the log file.
    """
    with open(log_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logging.info(f"Model summary saved to {log_file}")

# Timer function for measuring execution time
import time
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"Execution time: {self.interval:.4f} seconds")

# Example usage
if __name__ == '__main__':
    set_random_seed(42)
    logger = setup_logger(log_file='train.log')
    logger.info("Starting the project...")
    
    # Load config
    config = load_config(config_path='configs/config.yaml')
    
    # Timer example
    with Timer():
        # Your code for model training or evaluation goes here
        pass
