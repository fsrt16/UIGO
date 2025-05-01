

# 🧠 UIGO: Unified Inception-U-Net with Gravitational Optimization

**Full Title**: *Revolutionizing Liver Tumor Detection: The Novel Unified Inception-U-Net Hybrid Gravitational Optimization Model (UIGO) for Advanced Automated Medical Image Segmentation and Feature Selection*

---

## 📌 Overview

**UIGO** is a hybrid deep learning framework that integrates:
- **Inception modules** for multi-scale feature extraction,
- A **U-Net-inspired encoder–decoder structure** for semantic segmentation,
- **Gravitational Optimization** for intelligent feature selection.

This model is specifically designed to enhance segmentation performance in **liver tumor detection** from medical imaging (e.g., CT scans), enabling more accurate and efficient identification of affected regions.

---

## 🏗️ Architecture Highlights

- **Encoder**: Inception Blocks, Residual Connections, Channel Attention.
- **Bottleneck**: Multi-path dilated convolutions for contextual learning.
- **Decoder**: Symmetrical skip connections with upsampling and convolution.
- **Optimizer**: Gravitational Optimization Algorithm (GOA) to select and refine salient feature representations.

---

## 📁 Project Structure

UIGO/ ├── README.md ├── LICENSE ├── requirements.txt ├── .gitignore │ ├── data/ │ ├── raw/ # Original medical images and labels │ ├── processed/ # Preprocessed data (resized, normalized) │ └── utils.py # Data preprocessing, augmentation │ ├── models/ │ ├── uigo_net.py # Main hybrid model │ ├── blocks.py # Inception, attention, convolutional modules │ └── optimization.py # Gravitational Optimization Algorithm │ ├── training/ │ ├── train.py # Training loop, model fitting │ ├── loss_functions.py # Dice loss, focal loss, etc. │ ├── metrics.py # IOU, Dice, AUC, precision, recall │ └── scheduler.py # Learning rate schedulers and callbacks │ ├── evaluation/ │ ├── evaluate.py # Testing and final evaluation pipeline │ └── visualize.py # Prediction visualization and mask overlay │ ├── utils/ │ └── helpers.py # Logger, seed setting, config reader │ └── configs/ ├── config.yaml # Hyperparameters, paths, batch size └── dataset.yaml # Dataset splits and data sources

yaml
Copy
Edit

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UIGO.git
cd UIGO
```
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt
📊 Evaluation Metrics
UIGO is benchmarked using the following metrics:

Dice Coefficient

Intersection over Union (IoU)

Area Under Curve (AUC)

Sensitivity / Specificity

Hausdorff Distance (optional for clinical robustness)

🧪 Datasets
The model is evaluated on publicly available liver segmentation datasets:

LiTS (Liver Tumor Segmentation Challenge)

3DIRCADB (3D Image Reconstruction for Comparison of Algorithm Database)

Preprocessing includes resizing, histogram equalization, and data augmentation (flip, rotate, scale).

🚀 Training
bash
Copy
Edit
python training/train.py --config configs/config.yaml
Parameters like epochs, learning rate, and input size can be modified in the config.yaml file.

🔍 Results
UIGO consistently outperforms traditional U-Net variants in terms of segmentation accuracy, robustness, and sensitivity to tumor boundaries. Visualization tools enable side-by-side comparison of predictions with ground truths.

📄 License
This project is licensed under the MIT License. See LICENSE for more details.

👨‍🔬 Authors and Contributions
This work is developed as part of an advanced medical imaging research initiative. Contributions include model architecture design, gravitational optimization formulation, dataset preparation, and benchmarking.
