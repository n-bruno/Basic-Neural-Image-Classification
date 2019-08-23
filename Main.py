import tensorflow as tf
from numba.typing.builtins import Len
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from Constants import Constants

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Turns images into grayscale.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Here we define our layers
model = keras.Sequential([
    # Flatten data, so it's a 1D array, so we can pass it to each input neuron neuron
    keras.layers.Flatten(input_shape=(Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH)),
    # Hidden layer (layer between the input and output layer).
    # This gives our neural network more complexity.
    # Dense layer = fully connected layer.
    keras.layers.Dense(Constants.HIDDEN_LAYER_SIZE, activation="relu"),
    # Output layer. Each neuron represents the probability of the image being each class.
    # Softmax the output a probability (and adds up to one).
    keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epoch is a training cycle.
# https://stackoverflow.com/questions/31155388/meaning-of-an-epoch-in-neural-networks-training
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# What images should we predict.
predictions = model.predict(test_images)

# Console output whether the AI was right or not.
for i in range(len(class_names)):
    print('Image was {0}, Prediction {1}: AI was {2}'.format(
        class_names[test_labels[i]],
        class_names[np.argmax(predictions[i])],
        ('correct' if class_names[test_labels[i]] == class_names[np.argmax(predictions[i])] else 'WRONG'))
    )

# Chart output showing the probability of each scenario
for i in range(len(class_names)):
    plt.xlabel("Garments")
    plt.ylabel('Probability')
    plt.grid(True)
    plt.bar(x=np.arange(len(predictions[i])), height=predictions[i], align='center', alpha=0.5)
    plt.plot()
    plt.title(class_names[np.argmax(predictions[i])])

    plt.xticks(np.r_[0, 1:len(class_names)], class_names)

    # plt.imshow(test_images[i], cmap=plt.cm.binary, extent=[0, len(class_names), 0, 1])
    plt.show()