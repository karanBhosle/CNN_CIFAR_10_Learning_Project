# CNN_CIFR_10 — Learn Image Classification with CIFAR-10

This repository is a **teaching project**: you follow a Jupyter notebook that builds a convolutional neural network (CNN) step by step on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Before every code cell, a **markdown “Learning” section** explains concepts in plain language so you know what you are about to run and why it matters.

**Notebook:** [`CNN_CIFR_10/CNN_CIFR_10.ipynb`](CNN_CIFR_10/CNN_CIFR_10.ipynb)

---

## Who this is for

- Students or self-learners starting **deep learning for images**
- Data scientists who know Python but are new to **TensorFlow / Keras**
- Anyone who wants a **small, complete pipeline**: load data → preprocess → model → train → evaluate

**Prerequisites:** Basic Python (loops, functions), comfort with `pip install`, and optional familiarity with NumPy arrays. No prior CNN experience required.

---

## What you will build

You will train a classifier that assigns each 32×32 RGB image to one of ten classes:

| ID | Class       | ID | Class       |
|----|-------------|----|-------------|
| 0  | airplane    | 5  | dog         |
| 1  | automobile  | 6  | frog        |
| 2  | bird        | 7  | horse       |
| 3  | cat         | 8  | ship        |
| 4  | deer        | 9  | truck       |

The model is intentionally **small** so it runs on a laptop CPU in a few minutes. The focus is **understanding the workflow**, not state-of-the-art accuracy.

---

## How to use this repo (recommended path)

1. Read this README once for context.
2. Open the notebook and run cells **top to bottom**.
3. **Read each Learning markdown cell** before running the code under it.
4. After training, try the **“Ideas to try next”** section at the end of the notebook.

---

## End-to-end pipeline (concept map)

```text
CIFAR-10 images (0–255 pixels)
        │
        ▼  scale to [0, 1], one-hot labels
Preprocessed tensors
        │
        ▼  Conv2D → Pool → Conv2D → Pool → Dense → Softmax
CNN predictions (10 class probabilities)
        │
        ▼  fit() on training data, monitor validation split
Trained weights
        │
        ▼  evaluate on held-out test set + plot samples
Reported accuracy + visual error analysis
```

---

## Notebook sections and what you learn

| Section | You will understand… |
|--------|----------------------|
| **Libraries** | Which tools handle arrays, plots, and training |
| **Load data** | Train/test size, automatic download, integer labels |
| **Shapes** | `(samples, height, width, channels)` for images |
| **Labels** | Why Keras returns shape `(1,)` per label |
| **EDA grid** | Why you visualize data before modeling |
| **Preprocessing** | Pixel scaling, `float32`, and one-hot encoding for softmax |
| **CNN layers** | Convolution, pooling, flatten, dense, softmax roles |
| **Compile** | Adam, categorical cross-entropy, accuracy metric |
| **Baseline eval** | ~10% chance accuracy before training |
| **Training** | Epochs, batch size, `validation_split`, overfitting signals |
| **Test + plots** | Generalization vs. training, reading mistakes |

---

## Core concepts (short reference)

### Convolutional layer (`Conv2D`)

A set of **filters** scans the image and produces **feature maps** (e.g. edge-like or texture-like responses). Stacking conv layers lets the network build **hierarchical** features—from simple to more complex.

### Max pooling (`MaxPooling2D`)

Downsamples each feature map (e.g. 2×2 window → single max value). This **reduces size**, lowers compute, and adds some **robustness** to small shifts in the image.

### Softmax + categorical cross-entropy

The last layer outputs **10 probabilities** that sum to 1. The loss compares those probabilities to the **one-hot** true class. This is the standard setup for **multi-class single-label** classification.

### Train vs. validation vs. test

- **Training set:** used to update weights (`fit`).
- **Validation (here: 10% of train):** monitored during training to spot overfitting.
- **Test set:** used **once** at the end for an unbiased accuracy estimate. Do not tune hyperparameters on the test set.

---

## Model architecture (summary)

| Stage | Layer | Notes |
|-------|--------|--------|
| 1 | Conv2D, 32 filters, 3×3, ReLU | First feature extraction |
| 2 | MaxPooling2D, 2×2 | Spatial downsampling |
| 3 | Conv2D, 64 filters, 3×3, ReLU | Deeper features |
| 4 | MaxPooling2D, 2×2 | Further downsampling |
| 5 | Flatten | Vector for dense layers |
| 6 | Dense, 64, ReLU | Non-linear combination of features |
| 7 | Dense, 10, softmax | Class probabilities |

**Training defaults:** 5 epochs, batch size 64, Adam optimizer, 10% validation split.

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/karanBhosle/CNN_CIFAR_10_Learning_Project.git
cd CNN_CIFAR_10_Learning_Project
pip install tensorflow numpy matplotlib jupyter
```

### 2. Launch the notebook

```bash
jupyter notebook CNN_CIFR_10/CNN_CIFR_10.ipynb
```

On first run, `cifar10.load_data()` downloads the dataset (cached for later runs).

### 3. Google Colab (optional)

Upload the notebook or open the repo from Colab. Enable a **GPU** runtime to train faster; CPU is fine for learning.

---

## Learning outcomes

After completing the notebook, you should be able to:

1. Load and inspect a standard vision dataset in Keras.
2. Explain why pixels are scaled and labels are one-hot encoded.
3. Describe the purpose of conv, pool, dense, and softmax layers in a simple CNN.
4. Interpret training logs and a **random baseline** (~10% on CIFAR-10).
5. Report test accuracy and **inspect** misclassified images.

---

## Common mistakes (and how to avoid them)

| Mistake | What goes wrong | Fix |
|--------|------------------|-----|
| Skipping shape checks | Wrong `input_shape`, silent broadcasting bugs | Print `.shape` after load and preprocess |
| Forgetting `/255.0` | Slow or unstable training | Scale to `[0, 1]` |
| Integer labels with softmax | Loss/error or poor training | Use `to_categorical` |
| Tuning on test set | Overly optimistic accuracy | Tune on validation; test once at end |
| Only looking at accuracy | Miss class confusion (cat/dog) | Plot predictions |

---

## Ideas to extend the project

- Plot training vs. validation accuracy from `history.history`.
- Add **dropout** or **batch normalization** and compare validation curves.
- Apply **data augmentation** (flips, small rotations).
- Save the best checkpoint with `tf.keras.callbacks.ModelCheckpoint`.
- Experiment with a deeper network or transfer learning (e.g. MobileNet on resized images).

---

## Technologies

- [TensorFlow](https://www.tensorflow.org/) / Keras  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  

---

## About

**Author:** Karan Bhosle  
**LinkedIn:** [karanbhosle](https://www.linkedin.com/in/karanbhosle/)

This project is meant for learning and teaching. Feedback and pull requests are welcome.
