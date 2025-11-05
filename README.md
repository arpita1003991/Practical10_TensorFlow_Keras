**Aim**

To build and train a Machine Learning model using TensorFlow and Keras that recognizes handwritten digits from the MNIST dataset.

**Overview**

This mini-project demonstrates how deep learning can classify handwritten digits (0–9) from 28×28 pixel grayscale images.
The workflow includes data loading, preprocessing, model design, training, evaluation, and visualization of results.

**Technologies Used**

Python 3

TensorFlow 2.x

Keras API

Matplotlib (for plotting accuracy graphs)

Google Colab (for implementation)

**Steps to Run the Project**

Open Google Colab

Go to colab.research.google.com

Create a new notebook

Rename it as Practical10_TensorFlow_Keras.ipynb

Copy–paste the following code:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nModel Accuracy on Test Data: {acc:.2f}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("mnist_digit_model.h5")


Run all cells

Expected test accuracy: 0.97 – 0.99

Upload output files to GitHub

mnist_digit_model.h5

Practical10_TensorFlow_Keras.ipynb

README.md

.gitignore
