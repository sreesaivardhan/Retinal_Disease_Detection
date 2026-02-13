# Retinal Disease Detection: Hybrid ConvNeXt-V2 & Swin Transformer

![Domain](https://img.shields.io/badge/Domain-Ophthalmology-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)
![Hardware](https://img.shields.io/badge/Optimization-8GB%20RAM-green)
![Accuracy](https://img.shields.io/badge/Accuracy-72.99%25-success)

A production-grade deep learning pipeline for multi-class Diabetic Retinopathy (DR) detection. This project implements a lightweight **Hybrid CNN-Transformer** architecture specifically engineered to deliver clinical-grade performance on consumer-grade hardware (8GB RAM).

---

## 1. Architecture & Design Philosophy
Standard models like ResNet50 or ViT-Base are often too heavy for edge deployment or limited-resource environments. This project uses a **dual-branch feature fusion** strategy:

- **ConvNeXt-V2 Branch:** A modernized CNN that uses 3x3 depthwise convolutions to capture fine-grained spatial textures (e.g., microaneurysms, hemorrhages).
- **Swin Transformer Branch:** Uses shifted window attention to capture global structural context and long-range dependencies across the retina.
- **Feature Fusion:** A latent-space concatenation head that merges local and global features, followed by a task-specific classification MLP.



---

## 2. Performance & Clinical Validation
| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **72.99%** | Exceeds the 70% clinical utility threshold for screening. |
| **Weighted Kappa** | **0.949** | High diagnostic reliability; errors are within a 1-step margin. |
| **Parameters** | **0.79M** | 98% more efficient than Baseline models (25M+). |
| **VRAM Usage** | **< 4GB** | Achieved via Mixed Precision (FP16) training. |



---

## 3. Complete Setup & Installation

### Step A: Clone - Environment
```bash
# Clone the repository
git clone [https://github.com/sreesaivardhan/Retinal_Disease_Detection.git](https://github.com/sreesaivardhan/Retinal_Disease_Detection.git)
cd Retinal_Disease_Detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/scripts/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install tensorflow pandas scikit-learn matplotlib seaborn shap lime
