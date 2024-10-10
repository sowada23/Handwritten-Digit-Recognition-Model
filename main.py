import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize means all data would be in ragne between 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape data to add a channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train our model
# epoch means how many times deos the model see the same data
model.fit(x_train, y_train, epochs = 5)

# see the accuracy of the modelimport pandas as pdå
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
print(f"Loss: {loss:.2f}")

# Make predictions on new images
for i in range(0, 10):
    img = cv.imread(f'{i}.png', cv.IMREAD_GRAYSCALE)
    img = np.invert(np.array([img]))
    img = np.expand_dims(img, axis=-1)
    prediction = model.predict(img)
    print('-------------------')
    print("The predicted output is: ", np.argmax(prediction))
    print('-------------------')
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
    
