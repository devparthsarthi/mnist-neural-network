import gzip
import struct
import numpy as np


def load_images(filename):

    with gzip.open(filename, "rb") as f:

        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        if magic != 2051:
            raise ValueError("Invalid image file")

        data = f.read()

        images = np.frombuffer(
            data,
            dtype=np.uint8
        )

        images = images.reshape(
            num_images,
            rows,
            cols
        )

        # Flatten 28 × 28 → 784
        images = images.reshape(
            num_images,
            784
        )

        # Normalize 0–255 → 0–1
        images = images.astype(np.float32) / 255.0

        return images


def load_labels(filename):

    with gzip.open(filename, "rb") as f:

        magic, num_labels = struct.unpack(
            ">II",
            f.read(8)
        )

        if magic != 2049:
            raise ValueError("Invalid label file")

        data = f.read()

        labels = np.frombuffer(
            data,
            dtype=np.uint8
        )

        return labels


def load_mnist():

    train_images = load_images(
        "data/train-images-idx3-ubyte.gz"
    )

    train_labels = load_labels(
        "data/train-labels-idx1-ubyte.gz"
    )

    test_images = load_images(
        "data/t10k-images-idx3-ubyte.gz"
    )

    test_labels = load_labels(
        "data/t10k-labels-idx1-ubyte.gz"
    )

    return (
        train_images,
        train_labels,
        test_images,
        test_labels
    )