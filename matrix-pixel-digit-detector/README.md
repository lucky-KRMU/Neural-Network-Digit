# Matrix Pixel Digit Detector

A lightweight, single-neuron classifier implemented from scratch using Python and NumPy to detect the digit **4** represented within a 4x4 binary pixel matrix.

This project demonstrates the core mathematical principles of neural networks, including weighted sums, sigmoid activation, forward propagation, cost calculation, backpropagation, and gradient descent—all without using deep learning frameworks like TensorFlow or PyTorch.

---

## 🧠 Core Architecture

The model uses a single **Sigmoid Neuron** with the following specifications:
- **Inputs ($x$):** 16 elements (a flattened 4x4 matrix representing pixel intensities of 0 or 1).
- **Weights ($w$):** 16 values initialized randomly using a standard normal distribution.
- **Bias ($b$):** Initialized to 0.
- **Activation Function:** Sigmoid function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  where $z$ is the weighted sum:
  $$z = \sum (w_i \cdot x_i) + b = \mathbf{w} \cdot \mathbf{x} + b$$

---

## 📁 File Structure

- **[Neuron.py](file:///c:/Users/lenovo/Desktop/temp/matrix-pixel-digit-detector/Neuron.py):** Contains the primary class definition (`Neuron`), the training loop, gradient descent update rules, training datasets (`training_data` and `training_data_balanced`), and test predictions.

---

## 🛠️ Implementation Details

### The `Neuron` Class
The class defines the following primary methods:

1. **`__init__(self, inputs: int)`**: Initializes the neural weights ($\mathbf{w}$) using `np.random.randn` and bias ($b$) to 0.
2. **`weighted_sum_func(self, x: list) -> float`**: Calculates the dot product of the weights and flattened inputs, plus the bias ($\mathbf{w} \cdot \mathbf{x} + b$).
3. **`activation(self, x: list) -> float`**: Passes the weighted sum through the Sigmoid activation function to get a probability value between 0 (not a 4) and 1 (is a 4).
4. **`train(self, training_input: list, lr: float = 0.01, epochs: int = 10)`**: Trains the neuron using gradient descent. It loops over the epochs and training data, calculating prediction errors and adjusting weights and bias according to:
   $$\Delta w_i = -\text{lr} \cdot \delta \cdot x_i$$
   $$\Delta bias = -\text{lr} \cdot \delta$$
   where the gradient/delta ($\delta$) is derived from the squared error loss:
   $$\delta = \text{prediction} \cdot (1 - \text{prediction}) \cdot (\text{prediction} - \text{label})$$

---

## 📊 Training Datasets

The model provides two custom-curated training sets:
- **`training_data`**: A dataset containing matrices representing digit `4` (labeled as `1`) and various non-4 digits such as `0`, `1`, `2`, `3`, `5`, `6`, `7`, `8`, and `9` (labeled as `0`).
- **`training_data_balanced`**: A smaller, balanced dataset focusing on differentiating `4` from visually similar numbers like `8` and `9`.

### Example Matrix Representation for "4":
```python
[
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 1]
]
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.x
- NumPy package

To install NumPy, run:
```bash
pip install numpy
```

### Running the Script
Execute the script using:
```bash
python Neuron.py
```

### Example Script Execution Flow
The script initiates a `Neuron(16)`, runs an initial prediction on an untrained input matrix, trains it over `1000` epochs, prints the final weights and bias matrix, and performs a post-training evaluation.
