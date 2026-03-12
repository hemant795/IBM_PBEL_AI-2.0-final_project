# MNIST CNN Classifier

A convolutional neural network trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset using PyTorch. Achieves ~99% test accuracy after 10 epochs.

## Architecture

```
Conv2d(1 → 10, 5×5) → ReLU → MaxPool2d(2)
Conv2d(10 → 20, 5×5) → Dropout2d → ReLU → MaxPool2d(2)
Linear(320 → 50) → Dropout(0.5) → ReLU
Linear(50 → 10)  [raw logits]
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Batch size:** 100
- **Epochs:** 10

## Results

| Epoch | Test Accuracy |
| ----- | ------------- |
| 1     | ~97%          |
| 5     | ~98%          |
| 10    | ~99%          |

The best model weights are saved to `best_model.pth` during training.

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- torchvision
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Local

```bash
python mnist_cnn.py
```
### Google Colab

The script will:

1. Download the MNIST dataset (into `./data/` on first run)
2. Train the CNN for 10 epochs, printing loss and GPU memory stats each batch
3. Save the best model to `best_model.pth`
4. Display a 2×5 prediction grid from the test set (green = correct, red = wrong)

## GPU Support

CUDA is used automatically if available. The script prints GPU name, allocated, and cached memory at startup and after training.

## Project Structure

```
.
├── mnist_cnn.py       # Main training & evaluation script
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```