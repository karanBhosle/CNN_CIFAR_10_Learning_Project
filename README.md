# **CNN_CIFR_10 - Image Classification using CIFAR-10**

### **Project Overview:**
This project implements a Convolutional Neural Network (CNN) model for image classification using the CIFAR-10 dataset. The model classifies images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The code includes data preprocessing, model building, training, evaluation, and visualization.

### **Key Features:**
- **Data Visualization**: Displays a sample of images from the CIFAR-10 training set with labels.
- **Model Architecture**: The CNN model consists of convolutional layers for feature extraction and dense layers for classification.
- **Prediction and Visualization**: Visualizes the predictions on sample test images.
- **Model Evaluation**: Evaluates the model on the test set and prints accuracy.

### **Technologies Used:**
- **TensorFlow**: For building and training the CNN model.
- **Keras**: For simplifying the neural network layers and model building.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **CIFAR-10 Dataset**: A collection of 60,000 32x32 color images in 10 classes, used for training and testing.

### **Getting Started:**
1. **Install Dependencies**:
   Install the necessary Python packages by running the following command:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Download Dataset**:
   The CIFAR-10 dataset is automatically loaded from Keras when the script is run.

3. **Run the Code**:
   Execute the provided Python code in a Jupyter notebook or Google Colab environment.

### **Model Architecture:**
The CNN model consists of:
1. **Conv2D Layer (32 filters, 3x3 kernel)**: Extracts features from the input image.
2. **MaxPooling2D Layer (2x2 pool size)**: Reduces the dimensionality of the feature map.
3. **Conv2D Layer (64 filters, 3x3 kernel)**: Further extracts complex features.
4. **MaxPooling2D Layer (2x2 pool size)**: Further reduces the dimensionality.
5. **Flatten Layer**: Flattens the 3D data into a 1D vector for feeding into fully connected layers.
6. **Dense Layer (64 neurons)**: A fully connected layer to make decisions based on extracted features.
7. **Dense Layer (10 neurons with softmax activation)**: The output layer for classifying into one of the 10 classes.

### **Training the Model:**
- The model is compiled using the **Adam optimizer** and **categorical cross-entropy loss** function.
- Training is done for **5 epochs** with a **batch size of 64** and **10% validation split**.

### **Evaluating the Model:**
- The model is evaluated on the test set, and the **accuracy** is printed.
- A function is created to predict the class of sample test images and visualize the predictions.

### **Code Structure:**
The code is organized into the following sections:
1. **Import Libraries**: Includes TensorFlow, NumPy, Matplotlib, and Keras.
2. **Data Loading and Preprocessing**: Loads the CIFAR-10 dataset, reshapes, and normalizes the data.
3. **Model Building**: Defines the architecture of the CNN model.
4. **Model Training**: Trains the model on the training data.
5. **Model Evaluation**: Evaluates the model on the test data and displays the accuracy.
6. **Prediction and Visualization**: Makes predictions on individual test images and displays them with the predicted and actual labels.

### **Running the Code:**
- **Step 1**: Import the necessary libraries and load the CIFAR-10 dataset.
- **Step 2**: Preprocess the data by reshaping and normalizing the images.
- **Step 3**: Define the CNN architecture and compile the model.
- **Step 4**: Train the model on the training data.
- **Step 5**: Evaluate the model on the test data and display the accuracy.
- **Step 6**: Visualize the predictions on random test images.

### **Contact Information:**
- **Project Developed by**: Karan Bhosle
- **LinkedIn Profile**: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)

Feel free to reach out for questions or collaborations!
