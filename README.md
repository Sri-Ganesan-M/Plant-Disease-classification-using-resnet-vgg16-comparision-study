# 🌿 Plant Disease Detection Using EfficientNet-B3

This repository contains the implementation and supporting resources for the research paper:  
**"Plant Disease Detection Using EfficientNet-B3"**  
Authored by Dr. M. Poonkodi, Chandrakanth V, Sri Ganesan M, and Pavan Krishna.O  
Affiliated with VIT Chennai, School of Computer Science & Engineering.

📄 **[Read the Paper](https://drive.google.com/file/d/11-1mlRm4SEzKsaqvpcC4QaFlTMcRbjRa/view?usp=share_link)**

---

## 📌 Overview

Plant diseases pose a significant threat to agricultural productivity and global food security. Traditional methods for detecting plant diseases are time-consuming and prone to human error.  
Our project addresses this problem by employing **EfficientNet-B3**, a state-of-the-art convolutional neural network, to detect and classify 88 plant diseases using image data.

---

## 🔍 Highlights

- 📸 Dataset: **Plant Village Merged Dataset** with **79,086 images** across **88 disease classes**
- 🔬 Model: **EfficientNet-B3**, selected for its balance between accuracy and computational efficiency
- 🎯 Accuracy: Achieved **~98%** accuracy with **0.1 loss**
- 🧪 Techniques: Data augmentation, transfer learning, ReLU activation, categorical cross-entropy
- 📊 Metrics: Accuracy, Precision, Recall, F1-Score

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Image resizing, normalization
   - Data augmentation (flipping, rotation, cropping)

2. **Model Training**
   - Pretrained EfficientNet-B3 weights
   - Optimizer: Adam
   - Loss Function: Categorical Cross-Entropy

3. **Evaluation**
   - Tested on unseen data
   - Compared with traditional models like VGG-16
   - Demonstrated superior performance

---

## 📈 Results Summary

| Metric       | Score  |
|--------------|--------|
| Accuracy     | ~98%   |
| Precision    | High   |
| Recall       | High   |
| F1-Score     | High   |

---

## 📦 Technologies Used

- 🐍 Python
- 🧠 TensorFlow / Keras
- 🖼️ OpenCV / PIL
- 📊 Matplotlib / Seaborn

---

## 📚 Citation

If you find this work helpful, please cite:

```bibtex
@article{efficientnet_b3_plant_disease,
  title={Plant Disease Detection Using EfficientNet-B3},
  author={Poonkodi, M. and Chandrakanth, V. and Sri Ganesan, M. and Pavan Krishna, O.},
  journal={IEEE Conference Proceedings},
  year={20XX}
}
