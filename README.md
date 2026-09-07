# Handwritten Digit Classification with PyTorch

A PyTorch project for classifying handwritten digits from the MNIST dataset.

## Model

The project uses a fully connected neural network:

```text
784 → 256 → 128 → 64 → 10
```

with ReLU activations between the linear layers.

The 784 input features come from flattening each 28 × 28 MNIST image.

The 10 output neurons correspond to the digits 0 through 9.

## Dataset

MNIST handwritten digit dataset.

- Image size: 28 × 28 pixels
- Input features: 784
- Classes: 10
- Task: Multiclass classification

## Loss Function

`CrossEntropyLoss` is used for multiclass classification.

The model outputs raw logits, which are passed directly to the loss function:

```python
logits = model(images)
loss = loss_fn(logits, labels)
```

Softmax is not applied before `CrossEntropyLoss`.

For prediction, the class with the highest logit is selected:

```python
predictions = logits.argmax(dim=1)
```

## Optimizers

Different optimizers were tested under the same conditions:

- SGD
- SGD with Momentum
- Adam

The experiments showed differences in convergence, stability, and final accuracy. SGD achieved strong performance but showed more fluctuations in the loss curve. Adding momentum increased learning speed and accuracy at certain points but could also cause the model to overshoot and lose accuracy. Adam produced smoother training behavior.

## Technologies

- Python
- PyTorch
- Torchvision
- TorchMetrics
- NumPy
- Matplotlib

## Project Structure

```text
.
├── data/
├── script.ipynb
├── README.md
└── requirements.txt
```

## Future Work

- Compare different learning rates
- Experiment with deeper and wider networks
- Compare fully connected networks with CNNs
- Test the model on custom handwritten digits
