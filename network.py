import numpy as np


class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=10, output_size=10):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier-style initialization
        self.W1 = (
            np.random.randn(hidden_size, input_size)
            * np.sqrt(2.0 / input_size)
        )

        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = (
            np.random.randn(output_size, hidden_size)
            * np.sqrt(2.0 / hidden_size)
        )

        self.b2 = np.zeros((output_size, 1))


    # ------------------------------------------
    # RELU
    # ------------------------------------------

    def relu(self, Z):
        return np.maximum(0, Z)


    def relu_derivative(self, Z):
        return (Z > 0).astype(float)


    # ------------------------------------------
    # SOFTMAX
    # ------------------------------------------

    def softmax(self, Z):

        Z = Z - np.max(
            Z,
            axis=0,
            keepdims=True
        )

        exp_Z = np.exp(Z)

        return exp_Z / np.sum(
            exp_Z,
            axis=0,
            keepdims=True
        )


    # ------------------------------------------
    # FORWARD PROPAGATION
    # ------------------------------------------

    def forward(self, X):

        Z1 = self.W1 @ X + self.b1

        A1 = self.relu(Z1)

        Z2 = self.W2 @ A1 + self.b2

        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2


    # ------------------------------------------
    # LOSS
    # ------------------------------------------

    def loss(self, A2, labels):

        batch_size = labels.shape[0]

        probabilities = A2[
            labels,
            np.arange(batch_size)
        ]

        probabilities = np.clip(
            probabilities,
            1e-15,
            1.0
        )

        return -np.mean(np.log(probabilities))


    # ------------------------------------------
    # BACKPROPAGATION
    # ------------------------------------------

    def backward(self, X, labels, Z1, A1, Z2, A2):

        batch_size = X.shape[1]

        # One-hot encoded labels
        Y = np.zeros(
            (self.output_size, batch_size)
        )

        Y[
            labels,
            np.arange(batch_size)
        ] = 1

        # Output layer
        dZ2 = A2 - Y

        dW2 = (dZ2 @ A1.T) / batch_size

        db2 = np.sum(
            dZ2,
            axis=1,
            keepdims=True
        ) / batch_size

        # Hidden layer
        dA1 = self.W2.T @ dZ2

        dZ1 = (
            dA1 *
            self.relu_derivative(Z1)
        )

        dW1 = (dZ1 @ X.T) / batch_size

        db1 = np.sum(
            dZ1,
            axis=1,
            keepdims=True
        ) / batch_size

        return dW1, db1, dW2, db2


    # ------------------------------------------
    # UPDATE PARAMETERS
    # ------------------------------------------

    def update_parameters(
        self,
        dW1,
        db1,
        dW2,
        db2,
        learning_rate
    ):

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


    # ------------------------------------------
    # PREDICT
    # ------------------------------------------

    def predict(self, X):

        _, _, _, A2 = self.forward(X)

        return np.argmax(A2, axis=0)