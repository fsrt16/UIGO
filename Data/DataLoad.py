import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Enable logging for better visibility during execution
logging.basicConfig(level=logging.INFO)

# Sanity check for dataset integrity
def sanity_check(dataset_dir, image_size=(128, 128), show_sample_images=False):
    """
    Perform sanity checks to ensure data is correctly formatted.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        image_size (tuple): Desired image size for resizing (default is (128, 128)).
        show_sample_images (bool): Whether to show a few sample images (default is False).
        
    Returns:
        None
    """
    # Ensure the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found.")

    # List image files
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(image_files) == 0:
        raise ValueError("No images found in the dataset directory.")

    logging.info(f"Found {len(image_files)} image(s) in the dataset.")
    
    # Check that image files can be loaded and resized
    sample_images = []
    for idx, image_file in enumerate(image_files[:5]):  # Check 5 sample images
        image_path = os.path.join(dataset_dir, image_file)
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            sample_images.append(img_array)
        except Exception as e:
            logging.error(f"Error loading image {image_file}: {e}")
    
    if show_sample_images:
        # Show the first 5 sample images
        fig, axes = plt.subplots(1, 5, figsize=(15, 15))
        for i, ax in enumerate(axes):
            ax.imshow(sample_images[i].astype("uint8"))
            ax.axis('off')
        plt.show()

# Data loading and preprocessing function
def load_and_preprocess_data(dataset_dir, image_size=(128, 128), validation_split=0.2, batch_size=32):
    """
    Load and preprocess images for training.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        image_size (tuple): Desired image size for resizing (default is (128, 128)).
        validation_split (float): Fraction of data to reserve for validation (default is 0.2).
        batch_size (int): Number of samples per batch (default is 32).
        
    Returns:
        train_dataset: A `tf.data.Dataset` object for training data.
        val_dataset: A `tf.data.Dataset` object for validation data.
    """
    # Image data generator for augmentations and normalization
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize images to the range [0, 1]
        rotation_range=20,    # Random rotations
        width_shift_range=0.2,  # Random horizontal shifts
        height_shift_range=0.2,  # Random vertical shifts
        shear_range=0.2,      # Random shear transformations
        zoom_range=0.2,       # Random zoom
        horizontal_flip=True,  # Random horizontal flips
        fill_mode='nearest',   # Fill pixels that are shifted
        validation_split=validation_split
    )

    # Load and split the data into training and validation sets
    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # Or 'categorical' or 'input' based on your task
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # Or 'categorical' or 'input' based on your task
        subset='validation'
    )

    # Convert to tf.data.Dataset for better performance (optional)
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    return train_dataset, val_dataset

# Data augmentation example for custom images (for segmentation tasks)
def augment_image(image, label):
    """
    Custom augmentation for segmentation tasks.
    
    Args:
        image (tensor): Input image tensor.
        label (tensor): Corresponding label mask tensor.
        
    Returns:
        Augmented image and label pair.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    
    # Random vertical flip
    if np.random.rand() > 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    
    # Random rotation
    angle = np.random.uniform(-30, 30)  # Random rotation between -30 and 30 degrees
    image = tf.image.rot90(image, k=int(angle // 90))
    label = tf.image.rot90(label, k=int(angle // 90))

    # Normalize images
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Example usage of the data pipeline
if __name__ == "__main__":
    # Set your dataset directory path
    dataset_dir = "/path/to/your/dataset"
    
    # Perform sanity check
    sanity_check(dataset_dir, show_sample_images=True)
    
    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(dataset_dir)
    
    # Print the shapes of the first batch of data (sanity check)
    for images, labels in train_dataset.take(1):
        logging.info(f"Training batch shape: {images.shape}, Labels shape: {labels.shape}")
        
    for images, labels in val_dataset.take(1):
        logging.info(f"Validation batch shape: {images.shape}, Labels shape: {labels.shape}")
