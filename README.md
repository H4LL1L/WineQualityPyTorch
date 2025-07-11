# 🍷 Wine Quality Prediction with PyTorch Neural Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)](https://jupyter.org/)

This project develops a deep learning model using PyTorch library to predict red wine quality. It performs quality classification by analyzing the physicochemical properties of wines.

## 🎯 Project Objective

Quality control is crucial in the wine industry. This project aims to perform automatic quality assessment using the physicochemical properties of wines. The model classifies wines into two categories: "low quality" (0-5 points) and "high quality" (6+ points).

## 📊 Dataset Features

**Red Wine Quality Dataset** - UCI Machine Learning Repository

- **📈 Sample Count:** 1,599 red wine samples
- **🔢 Feature Count:** 11 physicochemical features + 1 quality score
- **🎯 Target Variable:** Quality score (0-10 scale, converted to binary classification)

### 🧪 Features

1. **Fixed Acidity** - Fixed acidity
2. **Volatile Acidity** - Volatile acidity
3. **Citric Acid** - Citric acid
4. **Residual Sugar** - Residual sugar
5. **Chlorides** - Chlorides
6. **Free Sulfur Dioxide** - Free sulfur dioxide
7. **Total Sulfur Dioxide** - Total sulfur dioxide
8. **Density** - Density
9. **pH** - pH value
10. **Sulphates** - Sulphates
11. **Alcohol** - Alcohol content

## 🏗️ Model Architecture

### 🧠 Neural Network Structure

- **Input Layer:** 11 neurons (number of features)
- **Hidden Layer 1:** 512 neurons + BatchNorm + ReLU + Dropout
- **Hidden Layer 2:** 256 neurons + BatchNorm + ReLU + Dropout
- **Output Layer:** 2 neurons (binary classification)

### ⚙️ Technical Features

- **🔥 Activation Function:** ReLU
- **📊 Normalization:** Batch Normalization
- **🚫 Regularization:** Dropout (0.3)
- **⚖️ Loss Function:** Weighted Cross Entropy Loss
- **🎯 Optimizer:** AdamW
- **📉 Learning Rate Scheduler:** ReduceLROnPlateau
- **⏹️ Early Stopping:** Patience=10

## 🚀 Usage

### 📋 Requirements

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install jupyter
```

### 💻 Running the Project

1. Clone the repository:

```bash
git clone https://github.com/H4LL1L/WineQualityPyTorch.git
cd WineQualityPyTorch
```

2. Open Jupyter Notebook:

```bash
jupyter notebook wineproject.ipynb
```

3. Run the notebook cells sequentially.

## 📈 Model Performance

### 🎯 Results

- **🔍 Test Accuracy:** ~80-85% (depending on random state)
- **⚡ Training Time:** ~50-100 epochs
- **📊 Validation Strategy:** 80/20 train-test split with stratification

### 📊 Visualizations

- **📈 Learning Curves:** Training and validation loss/accuracy graphs
- **🔥 Feature Distribution:** Feature distributions before/after normalization
- **🎯 Confusion Matrix:** Model prediction performance

## 🛠️ Key Techniques

### 🔧 Data Preprocessing

- **📊 StandardScaler:** Z-score normalization (μ=0, σ=1)
- **⚖️ Stratified Split:** Data splitting while preserving class balance
- **🔢 Binary Classification:** Quality scores 6+ as 1, below 6 as 0

### 🎓 Model Optimization

- **🏋️ Weight Initialization:** Xavier Uniform
- **⚖️ Class Weighting:** Weighting for imbalanced class distribution
- **🎯 Hyperparameter Tuning:** Batch size, learning rate, dropout optimization
- **🔄 Learning Rate Scheduling:** Adaptive learning rate

## 📁 File Structure

```
WineQualityPyTorch/
├── 📓 wineproject.ipynb      # Main Jupyter Notebook
├── 📊 winequality-red.csv    # Dataset
├── 📖 README.md              # This file
└── 🤖 best_model.pth         # Trained model (created after running)
```

## 📄 License

This project is licensed under the MIT License.

## 📚 References

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [PyTorch Documentation](https://pytorch.org/docs/)
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, 2009.
