import numpy as np

from network import NeuralNetwork
from load_mnist import load_mnist


# ==========================================
# SETTINGS
# ==========================================

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01


# ==========================================
# LOAD DATA
# ==========================================

print("Loading MNIST...")

(
    train_images,
    train_labels,
    test_images,
    test_labels
) = load_mnist()

print("Training images:", train_images.shape)
print("Training labels:", train_labels.shape)

print("Test images:", test_images.shape)
print("Test labels:", test_labels.shape)


# ==========================================
# CREATE NETWORK
# ==========================================

network = NeuralNetwork(
    input_size=784,
    hidden_size=10,
    output_size=10
)

print("\nNetwork:")
print("784 -> 10 -> 10")


# ==========================================
# TRAINING
# ==========================================

num_samples = train_images.shape[0]

for epoch in range(EPOCHS):

    # Shuffle dataset
    permutation = np.random.permutation(
        num_samples
    )

    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    total_loss = 0
    total_correct = 0

    num_batches = 0


    for start in range(
        0,
        num_samples,
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        batch_images = train_images[start:end]
        batch_labels = train_labels[start:end]

        # (batch, 784)
        # ↓
        # (784, batch)

        X = batch_images.T


        # Forward
        Z1, A1, Z2, A2 = network.forward(X)


        # Loss
        loss = network.loss(
            A2,
            batch_labels
        )


        # Predictions
        predictions = np.argmax(
            A2,
            axis=0
        )


        # Accuracy
        correct = np.sum(
            predictions == batch_labels
        )


        # Backpropagation
        dW1, db1, dW2, db2 = network.backward(
            X,
            batch_labels,
            Z1,
            A1,
            Z2,
            A2
        )


        # Update weights
        network.update_parameters(
            dW1,
            db1,
            dW2,
            db2,
            LEARNING_RATE
        )


        total_loss += loss
        total_correct += correct
        num_batches += 1


    # ======================================
    # EPOCH RESULTS
    # ======================================

    average_loss = (
        total_loss /
        num_batches
    )

    accuracy = (
        total_correct /
        num_samples
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ==========================================
# TESTING
# ==========================================

print("\n==============================")
print("TESTING")
print("==============================")


test_correct = 0

test_samples = test_images.shape[0]


for start in range(
    0,
    test_samples,
    BATCH_SIZE
):

    end = start + BATCH_SIZE

    batch_images = test_images[start:end]
    batch_labels = test_labels[start:end]

    X = batch_images.T

    predictions = network.predict(X)

    test_correct += np.sum(
        predictions == batch_labels
    )


test_accuracy = (
    test_correct /
    test_samples
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")


X = test_images[:20].T

predictions = network.predict(X)


for i in range(20):

    print(
        f"Image {i + 1:2d} | "
        f"Actual: {test_labels[i]} | "
        f"Predicted: {predictions[i]}"
    )

    import matplotlib.pyplot as plt


def visualize_network(network, image, label):

    X = image.reshape(784, 1)

    Z1, A1, Z2, A2 = network.forward(X)

    prediction = np.argmax(A2)

    loss = network.loss(A2, np.array([label]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ==========================================
    # INPUT IMAGE
    # ==========================================

    axes[0].imshow(
        image.reshape(28, 28),
        cmap="gray"
    )

    axes[0].set_title(
        f"Input Image\nActual: {label}"
    )

    axes[0].axis("off")


    # ==========================================
    # HIDDEN LAYER
    # ==========================================

    axes[1].bar(
        range(10),
        A1[:, 0]
    )

    axes[1].set_title(
        "Hidden Layer Activations"
    )

    axes[1].set_xlabel(
        "Neuron"
    )

    axes[1].set_ylabel(
        "Activation"
    )

    axes[1].set_xticks(range(10))


    # ==========================================
    # OUTPUT LAYER
    # ==========================================

    axes[2].bar(
        range(10),
        A2[:, 0]
    )

    axes[2].set_title(
        f"Output\nPrediction: {prediction}"
    )

    axes[2].set_xlabel(
        "Digit"
    )

    axes[2].set_ylabel(
        "Probability"
    )

    axes[2].set_xticks(range(10))

    axes[2].set_ylim(0, 1)


    fig.suptitle(
        f"Neural Network: 784 -> 10 -> 10 | Loss: {loss:.4f}"
    )

    plt.tight_layout()
    plt.show()