# Deep Learning Practice with PyTorch & Keras

This repository contains hands-on notebooks where I practice and experiment with core deep learning concepts using **PyTorch** and **Keras**. 
Each notebook explores a specific topic, ranging from activation functions and gradient descent to convolutional networks and transformers.

### Fundamentals
- **Activation Function with PyTorch.ipynb** – Visualizes and implements key activation functions (ReLU, Sigmoid, Tanh).
- **Activation with Dataset.ipynb** – Applies activation functions in a small model with synthetic data.
- **Mini batch gradient descent.ipynb** – Explores mini-batch updates for optimization.
- **Stochastic Gradient Descent.ipynb** – Demonstrates SGD and its convergence behavior.
- **Momentum with Different Polynomials.ipynb** – Shows how momentum affects optimization on different loss curves.
- **He Initialization.ipynb** – Tests He initialization with ReLU networks.
- **Test Xavier & Uniform Initialization on MNIST.ipynb** – Compares initialization strategies using the MNIST dataset and `tanh` activation.

### Neural Networks
- **Neural Network with One Hidden Layer.ipynb** – A basic feedforward neural net from scratch.
- **Multiclass Neural Network.ipynb** – A model to classify multiple output classes.
- **Batch Normalization.ipynb** – Applies batch normalization and analyzes its effects.

### Architectures
- **CNNs Worksheet with Keras.ipynb** – Convolutional neural network built using Keras.
- **Convolution Network.ipynb** – Manual CNN implementation and visualization.
- **Custom Layer with Keras.ipynb** – Shows how to build and integrate custom Keras layers.

### Advanced Topics
- **Auto Encoder.ipynb** – Builds a simple autoencoder for dimensionality reduction.
- **Advance Transformer.ipynb** – Early experimentation with transformer architecture components.

### Worksheets
- **Keras Worksheet.ipynb** – Covers model training and evaluation using Keras.
- **Dropout Validation.ipynb** *(not listed above but might exist)* – Likely explores dropout regularization.

---

## 🛠 Requirements
- Python 3.8–3.10
- PyTorch
- TensorFlow / Keras
- NumPy, Matplotlib, scikit-learn

Install all dependencies:
```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install numpy matplotlib scikit-learn
