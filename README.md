# MNIST Neural Network

A neural network built from scratch in Python for handwritten digit classification using the MNIST dataset.

## Overview

This project implements the core components of a feed-forward neural network without relying on high-level deep learning frameworks. It is intended to demonstrate how neural networks work internally, including data loading, forward propagation, activation functions, loss calculation, backpropagation, and parameter updates.

## Features

- Loads and processes the MNIST handwritten digit dataset
- Implements a neural network from scratch
- Forward propagation
- Backpropagation
- Gradient-based parameter updates
- Digit classification from 0–9
- Simple Python-based implementation for learning and experimentation

## Project Structure

```text
mnist-neural-network/
├── load_mnist.py       # MNIST data loading utilities
├── network.py          # Neural network implementation
├── main.py             # Program entry point / training and evaluation
├── download_mnist.py   # Downloads the MNIST dataset
└── .gitignore          # Files excluded from Git
```

> File names may differ depending on the current version of the project.

## Dataset

The project uses the **MNIST handwritten digit dataset**, containing grayscale images of handwritten digits from 0 to 9.

The dataset files are intentionally excluded from the repository because they are generated/downloaded locally and do not need to be stored in Git.

## Requirements

- Python 3.x
- NumPy

Install the dependency with:

```bash
pip install numpy
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/devparthsarthi/mnist-neural-network.git
cd mnist-neural-network
```

Download or prepare the MNIST dataset using the project's download script, then run the main program:

```bash
python download_mnist.py
python main.py
```

If your local project uses a different entry point, run the corresponding Python file instead.

## How It Works

The network follows the standard machine-learning pipeline:

1. Load the MNIST images and labels.
2. Preprocess the image data into numerical input vectors.
3. Pass the inputs through the neural network using forward propagation.
4. Calculate the prediction error using a loss function.
5. Use backpropagation to calculate gradients.
6. Update the network parameters using gradient descent.
7. Evaluate the trained model on unseen test images.

## Learning Goals

This project is primarily educational. Building the network without a high-level framework helps understand the mathematics and implementation behind neural networks instead of treating them as a black box.

## Future Improvements

- Add configurable network architecture
- Add training and validation metrics
- Improve training performance
- Add visualization of predictions
- Save and load trained model parameters
- Experiment with different activation and optimization methods

## Author

**Parthsarthi Sharma**

GitHub: [@devparthsarthi](https://github.com/devparthsarthi)
