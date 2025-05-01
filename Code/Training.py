import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. Data Loading (replace this with your own data loader)
# X: image inputs, Y: binary masks
# Example: X.shape = (1000, 256, 256, 1), Y.shape = (1000, 256, 256, 1)
# X, Y = load_dataset()

# 2. Hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
INPUT_SHAPE = X.shape[1:]  # e.g., (256, 256, 1)
MODEL_NAME = 'tnet_model'
BASE_PATH = './model_checkpoints'

# 3. Create checkpoint directory
os.makedirs(BASE_PATH, exist_ok=True)

# 4. Model building function
def get_model(input_shape):
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)

    # Example: simple U-Net structure
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 5. Setup Callbacks
def get_callbacks(fold):
    checkpoint_path = os.path.join(BASE_PATH, f"{MODEL_NAME}_fold{fold+1}.h5")
    return [
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

# 6. Training with K-Fold Cross-Validation
from sklearn.model_selection import KFold

NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n📁 Starting Fold {fold + 1}/{NUM_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    model = get_model(INPUT_SHAPE)

    callbacks = get_callbacks(fold)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Optional: Save training history
    history_path = os.path.join(BASE_PATH, f"history_fold{fold+1}.npy")
    np.save(history_path, history.history)

print("\n✅ Training complete for all folds.")
