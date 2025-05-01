import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import tensorflow as tf

# ----------------------------
# Utility Metrics
# ----------------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ----------------------------
# Evaluate the Model
# ----------------------------

def evaluate_model(model, X_test, Y_test, threshold=0.5, num_classes=1):
    preds = model.predict(X_test)
    
    if num_classes == 1:
        preds_binary = (preds > threshold).astype(np.uint8)
    else:
        preds_binary = np.argmax(preds, axis=-1)
        Y_test = np.argmax(Y_test, axis=-1)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'dice': [],
        'iou': []
    }

    for i in range(len(X_test)):
        y_true = Y_test[i].astype(np.uint8).flatten()
        y_pred = preds_binary[i].astype(np.uint8).flatten()

        metrics['accuracy'].append(np.mean(y_true == y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=1))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=1))
        metrics['f1_score'].append(f1_score(y_true, y_pred, zero_division=1))
        metrics['dice'].append(dice_coefficient(y_true, y_pred))
        metrics['iou'].append(iou_score(y_true, y_pred))

    # Print Average Results
    for key in metrics:
        print(f"{key.upper()}: {np.mean(metrics[key]):.4f}")

    return preds_binary

# ----------------------------
# Plot Predictions vs Ground Truth
# ----------------------------

def plot_predictions(X, Y_true, Y_pred, index=0):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(X[index].squeeze(), cmap='gray')
    axs[0].set_title("Input Image")

    axs[1].imshow(Y_true[index].squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth")

    axs[2].imshow(Y_pred[index].squeeze(), cmap='gray')
    axs[2].set_title("Predicted Mask")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Confusion Matrix
# ----------------------------

def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ----------------------------
# Example Usage
# ----------------------------

# Assuming X_test and Y_test are already loaded and preprocessed
# model = ... (your trained T-Net / Unet / Vnet)
# X_test, Y_test = ... (your data)

# Evaluate
predicted_masks = evaluate_model(model, X_test, Y_test, threshold=0.5)

# Visualize Predictions
plot_predictions(X_test, Y_test, predicted_masks, index=5)  # Change index for different samples

# Show Confusion Matrix
show_confusion_matrix(Y_test, predicted_masks)
