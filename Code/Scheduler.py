import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import logging

# Learning Rate Scheduler (e.g., Step Decay, Exponential Decay, etc.)
def step_decay_scheduler(initial_lr=1e-3, drop_factor=0.5, epoch_drop=10):
    """
    Creates a step decay learning rate scheduler.
    
    Args:
        initial_lr (float): Initial learning rate (default is 1e-3).
        drop_factor (float): Factor by which the learning rate will be reduced.
        epoch_drop (int): Number of epochs after which the learning rate drops.
    
    Returns:
        function: A function that takes an epoch as input and returns the learning rate.
    """
    def scheduler(epoch):
        lr = initial_lr * (drop_factor ** (epoch // epoch_drop))
        logging.info(f"Epoch {epoch}: Learning rate adjusted to {lr}")
        return lr
    
    return scheduler

# Exponential Decay Scheduler
def exponential_decay_scheduler(initial_lr=1e-3, decay_rate=0.96, decay_steps=100000):
    """
    Creates an exponential decay learning rate scheduler.
    
    Args:
        initial_lr (float): Initial learning rate (default is 1e-3).
        decay_rate (float): Rate at which the learning rate decays (default is 0.96).
        decay_steps (int): Number of steps before applying decay.
    
    Returns:
        function: A function that takes an epoch as input and returns the learning rate.
    """
    def scheduler(epoch):
        lr = initial_lr * decay_rate ** (epoch / decay_steps)
        logging.info(f"Epoch {epoch}: Learning rate adjusted to {lr}")
        return lr
    
    return scheduler

# Custom Learning Rate Scheduler (Cosine Annealing Scheduler)
class CosineAnnealingScheduler(Callback):
    """
    Custom Learning Rate Scheduler that adjusts the learning rate following a cosine annealing schedule.
    """
    def __init__(self, initial_lr, T_max, eta_min=1e-6):
        """
        Initialize the cosine annealing scheduler.
        
        Args:
            initial_lr (float): The starting learning rate.
            T_max (int): The number of iterations or epochs for the annealing process.
            eta_min (float): The minimum learning rate (default is 1e-6).
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def on_epoch_begin(self, epoch, logs=None):
        """
        This callback method will be triggered at the start of each epoch.
        
        Args:
            epoch (int): The current epoch.
            logs (dict): The logs for the epoch.
        """
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + tf.cos(tf.constant(epoch / self.T_max * 3.1416))) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        logging.info(f"Epoch {epoch}: Learning rate adjusted to {lr.numpy()}")

# Learning Rate Scheduler Callback for TensorFlow/Keras
def create_lr_scheduler_callback(scheduler_type="step_decay", **kwargs):
    """
    Creates a learning rate scheduler callback based on the given scheduler type.
    
    Args:
        scheduler_type (str): Type of scheduler to use ('step_decay', 'exponential_decay', 'cosine_annealing').
        **kwargs: Additional arguments for the scheduler.
    
    Returns:
        callback (tf.keras.callbacks.Callback): The learning rate scheduler callback.
    """
    if scheduler_type == "step_decay":
        return LearningRateScheduler(step_decay_scheduler(**kwargs))
    elif scheduler_type == "exponential_decay":
        return LearningRateScheduler(exponential_decay_scheduler(**kwargs))
    elif scheduler_type == "cosine_annealing":
        return CosineAnnealingScheduler(**kwargs)
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported")

# Example usage: Add scheduler to model training
if __name__ == "__main__":
    # Create a sample model for demonstration
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Create scheduler callback
    lr_scheduler = create_lr_scheduler_callback(scheduler_type="step_decay", 
                                                initial_lr=1e-3, drop_factor=0.5, epoch_drop=10)
    
    # Train the model with the scheduler
    model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[lr_scheduler])
