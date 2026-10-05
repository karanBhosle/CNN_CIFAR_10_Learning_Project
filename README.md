# CNN_CIFR_10 — Image Classification on CIFAR-10

A hands-on Jupyter notebook that trains a convolutional neural network (CNN) on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The notebook is structured for learning: **each code cell includes short “Learning” comments** that explain why that step matters, not just what the code does.

## Project overview

CIFAR-10 contains 60,000 color images (32×32) in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project covers:

- Loading and exploring the dataset  
- Preprocessing (scaling and one-hot labels)  
- Building a small CNN in Keras  
- Training with validation split  
- Evaluating on the test set and visualizing predictions  

## Notebook walkthrough

| Step | Topic | What you learn |
|------|--------|----------------|
| Imports | Libraries | How NumPy, Matplotlib, and Keras fit together for vision workflows |
| Load data | `cifar10.load_data()` | Train/test split and label format |
| Shapes | Tensor dimensions | `(N, H, W, C)` for images and `(N, 1)` for labels |
| Sample label | Indexing | Integer class IDs and `label[i][0]` |
| Plot grid | EDA | Why you visualize labels before training |
| Preprocess | `/255` + `to_categorical` | Normalization and matching loss to softmax output |
| Model | Conv → pool → dense | Feature extraction vs. classification layers |
| Compile | Adam + cross-entropy | Choosing optimizer, loss, and metrics |
| Baseline eval | Untrained accuracy | ~10% random baseline for 10 classes |
| Train | `fit()` | Epochs, batch size, and `validation_split` |
| Test + viz | Metrics + plots | Generalization and qualitative error analysis |

Open the notebook: [`CNN_CIFR_10/CNN_CIFR_10.ipynb`](CNN_CIFR_10/CNN_CIFR_10.ipynb).

## Model architecture (summary)

1. **Conv2D** (32 filters, 3×3, ReLU) + **MaxPooling2D** (2×2)  
2. **Conv2D** (64 filters, 3×3, ReLU) + **MaxPooling2D** (2×2)  
3. **Flatten** → **Dense** (64, ReLU) → **Dense** (10, softmax)  

Training defaults in the notebook: **5 epochs**, **batch size 64**, **10% validation split**, **Adam** optimizer, **categorical cross-entropy** loss.

## Getting started

1. **Install dependencies**

   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

2. **Run the notebook**

   ```bash
   jupyter notebook CNN_CIFR_10/CNN_CIFR_10.ipynb
   ```

   CIFAR-10 is downloaded automatically the first time you call `cifar10.load_data()`.

3. **Optional: Google Colab**  
   Upload the notebook or clone this repo and run cells top to bottom. A GPU runtime speeds up training but is not required for this small model.

## Key learnings (repository focus)

- **Inspect data first** — shapes, labels, and a few plotted images prevent silent bugs later.  
- **Preprocessing matches the model** — float pixels in `[0, 1]` and one-hot labels pair with softmax + categorical cross-entropy.  
- **CNNs exploit spatial structure** — convolutions detect local patterns; pooling reduces spatial size and cost.  
- **Baseline before training** — untrained test accuracy near 10% confirms the setup before you invest in training time.  
- **Validate while training** — a held-out slice of training data helps spot overfitting early.  
- **Metrics + visuals** — accuracy on the test set plus sample predictions give a fuller picture than a single number.  

## Technologies

- TensorFlow / Keras  
- NumPy  
- Matplotlib  

## Contact

**Karan Bhosle** — [LinkedIn](https://www.linkedin.com/in/karanbhosle/)

Questions and collaboration welcome.
