import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Number of folds
NUM_FOLDS = 5

# Placeholder lists to store fold results
all_metrics = {
    "fold": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "dice": [],
    "iou": []
}

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

# Load your dataset
# X, Y = ...  # X.shape = (N, H, W, C), Y.shape = (N, H, W, 1)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, X_val = X[train_index], X[test_index]
    Y_train, Y_val = Y[train_index], Y[test_index]

    # Model setup (assume get_model returns a fresh compiled model)
    model = get_model()
    model.fit(X_train, Y_train, epochs=10, batch_size=8, verbose=1, validation_data=(X_val, Y_val))

    preds = model.predict(X_val)
    preds_binary = (preds > 0.5).astype(np.uint8)

    for i in range(len(Y_val)):
        y_true = Y_val[i].astype(np.uint8).flatten()
        y_pred = preds_binary[i].astype(np.uint8).flatten()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=1)
        rec = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        dice = dice_coefficient(y_true, y_pred)
        iou = iou_score(y_true, y_pred)

        all_metrics['fold'].append(fold + 1)
        all_metrics['accuracy'].append(acc)
        all_metrics['precision'].append(prec)
        all_metrics['recall'].append(rec)
        all_metrics['f1_score'].append(f1)
        all_metrics['dice'].append(dice)
        all_metrics['iou'].append(iou)

# Convert to DataFrame
import pandas as pd
results_df = pd.DataFrame(all_metrics)
print(results_df.groupby('fold').mean())

# ----------------------------
# 📊 Statistical Analysis
# ----------------------------

# Summary stats
print("\nOverall Summary:")
summary = results_df.drop('fold', axis=1).agg(['mean', 'std', 'min', 'max'])
print(summary)

# Boxplot per metric
plt.figure(figsize=(14, 6))
sns.boxplot(data=results_df.drop(columns="fold"))
plt.title("Cross-Fold Evaluation Metrics")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
