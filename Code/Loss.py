import tensorflow as tf
from tensorflow.keras import backend as K

# Smooth factor to avoid division by zero
SMOOTH = 1e-6

# ===================== Dice Loss =====================
def dice_loss(y_true, y_pred):
    """
    Dice loss function to measure the overlap between the predicted and true masks
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

# ===================== Tversky Loss =====================
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Tversky Loss function is sensitive to class imbalance. It is used in segmentation tasks with imbalanced classes.
    alpha controls the weight for false positives, and beta controls the weight for false negatives.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    true_pos = K.sum(y_true_f * y_pred_f)
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    
    return 1 - (true_pos + SMOOTH) / (true_pos + alpha * false_pos + beta * false_neg + SMOOTH)

# ===================== Focal Loss =====================
def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss function to address class imbalance by down-weighting the loss for well-classified examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        epsilon = K.epsilon()
        y_pred_f = K.clip(y_pred_f, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true_f * K.log(y_pred_f)
        loss = alpha * K.pow((1 - y_pred_f), gamma) * cross_entropy
        
        return K.sum(loss)
    
    return focal_loss_fixed

# ===================== Combined Loss (BCE + Dice) =====================
def bce_dice_loss(y_true, y_pred):
    """
    Combined loss function with Binary Cross-Entropy and Dice Loss.
    This helps to combine the robustness of BCE for classification with the effectiveness of Dice for segmentation overlap.
    """
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    
    return bce_loss + dice_loss_val

# ===================== Combo Loss (BCE + Tversky) =====================
def bce_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Combined loss function with Binary Cross-Entropy and Tversky Loss.
    This is useful for segmentation tasks with class imbalance.
    """
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    tversky_loss_val = tversky_loss(y_true, y_pred, alpha, beta)
    
    return bce_loss + tversky_loss_val

# ===================== Mean Squared Error (MSE) =====================
def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss function.
    This is typically used in regression but can be used in segmentation if you want to treat the mask as a regression task.
    """
    return K.mean(K.square(y_true - y_pred))

# ===================== Weighted Cross-Entropy Loss =====================
def weighted_bce_loss(y_true, y_pred, pos_weight=1.0):
    """
    Weighted Binary Cross-Entropy Loss to address class imbalance by giving higher weights to the minority class.
    """
    epsilon = K.epsilon()
    y_pred_f = K.clip(y_pred, epsilon, 1. - epsilon)
    
    cross_entropy = -y_true * K.log(y_pred_f) - (1 - y_true) * K.log(1 - y_pred_f)
    loss = pos_weight * y_true * cross_entropy + (1 - y_true) * cross_entropy
    
    return K.mean(loss)

# ===================== Dice + Focal Loss =====================
def dice_focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    """
    A combination of Dice Loss and Focal Loss.
    The purpose of this loss function is to combine the advantages of both functions for segmentation tasks.
    """
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(gamma, alpha)(y_true, y_pred)
    
    return dice + focal

# ===================== Jaccard/IoU Loss =====================
def iou_loss(y_true, y_pred):
    """
    Intersection over Union (IoU) loss function.
    IoU is a measure of how well the predicted mask overlaps with the true mask.
    """
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (intersection + SMOOTH) / (union + SMOOTH)

# ===================== Soft Dice Loss =====================
def soft_dice_loss(y_true, y_pred):
    """
    Soft Dice Loss is a differentiable approximation of the Dice score, useful for segmentation tasks.
    """
    intersection = K.sum(y_true * y_pred)
    return 1 - (2. * intersection + SMOOTH) / (K.sum(y_true) + K.sum(y_pred) + SMOOTH)

# ===================== Weighted Dice Loss =====================
def weighted_dice_loss(y_true, y_pred, weights):
    """
    Weighted Dice Loss for class-imbalance, where each class is given a weight to focus the loss more on certain regions.
    """
    intersection = K.sum(y_true * y_pred * weights)
    return 1 - (2. * intersection + SMOOTH) / (K.sum(y_true * weights) + K.sum(y_pred * weights) + SMOOTH)

# ===================== Dice + Tversky Loss =====================
def dice_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    A combined loss function of Dice and Tversky loss.
    This loss function is especially helpful in dealing with segmentation tasks where imbalanced classes exist.
    """
    dice = dice_loss(y_true, y_pred)
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    
    return dice + tversky

# ===================== Gradient Weighted Loss =====================
def gradient_weighted_loss(y_true, y_pred):
    """
    A loss function that incorporates gradients to weigh the loss function more heavily on the edges of predicted masks.
    """
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    
    grad_loss = K.mean(K.square(grad_true - grad_pred))
    
    return grad_loss + dice_loss(y_true, y_pred)

# ===================== Asymmetric Loss =====================
def asymmetric_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Asymmetric loss function, which is useful when false positives and false negatives have different importance in segmentation.
    """
    true_pos = K.sum(y_true * y_pred)
    false_pos = K.sum((1 - y_true) * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    
    return alpha * false_pos + beta * false_neg + (1 - alpha - beta) * true_pos

# ===================== Custom Combined Loss (Dice + BCE + Focal) =====================
def custom_combined_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    A custom combined loss function with Dice, Binary Cross-Entropy, and Focal Loss.
    This function aims to capture both the overlap between the segmentation and the class imbalance.
    """
    bce_loss_val = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    focal_loss_val = focal_loss(gamma, alpha)(y_true, y_pred)
    
    return bce_loss_val + dice_loss_val + focal_loss_val

# ===================== Final Composite Loss =====================
def final_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0, pos_weight=1.0):
    """
    A final loss function that incorporates multiple loss types.
    Here we combine Dice Loss, Tversky Loss, Focal Loss, and BCE Loss for a composite approach.
    """
    bce = weighted_bce_loss(y_true, y_pred, pos_weight)
    dice = dice_loss(y_true, y_pred)
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    focal = focal_loss(gamma)(y_true, y_pred)
    
    return bce + dice + tversky + focal
