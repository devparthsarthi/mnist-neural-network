import os
import urllib.request

DATA_DIR = "data"

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

os.makedirs(DATA_DIR, exist_ok=True)

for file in FILES:

    url = BASE_URL + file
    path = os.path.join(DATA_DIR, file)

    if os.path.exists(path):
        print(f"{file} already exists. Skipping.")
        continue

    print(f"Downloading {file}...")

    urllib.request.urlretrieve(url, path)

    print(f"Downloaded: {file}")

print("\nMNIST dataset downloaded successfully!")