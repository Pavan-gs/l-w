import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers

# CNN --> Convolve / Convolution

# Feature detector/Kernel/Filter --> Weights

# Convolution --> Linear operation where multiplication of a set of weights with the input data takes place

# Kernel/Feature map : Output of the one filter applied to the previous layer
# Max pooling, Flattening, Zero padding, Fully connected layer

# Channels --> RGB (3), Grayscale (1)
# Size --> 28*28, 32*32, 64*64

# An image is between 0-255 pixels

from keras.datasets import mnist, fashion_mnist, cifar10
c = cifar10.load_data()

(xtrain,ytrain),(xtest,ytest) = cifar10.load_data()

for i in range(9):
    plt.subplot(2,5,i+1)
    plt.imshow(xtrain[i],cmap=plt.get_cmap('gray'))
    plt.show()
    
# Pre-processing
from keras.utils import to_categorical
ytrain = to_categorical((ytrain))
ytest = to_categorical(ytest)

train_norm = xtrain.astype('float32')
test_norm =xtest.astype('float32')

train_norm = train_norm/255.0
test_norm = test_norm/255.0

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
m2 = model.fit(train_norm,ytrain, epochs=10, batch_size=25, validation_data=(test_norm,ytest))

pred = model.predict(test_norm)

from sklearn.metrics import confusion_matrix, accuracy_score
pred[0]
np.round(pred[0],2)

# Plot the training and validation loss
plt.figure(figsize=(12, 5))

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(m2.history['loss'], label='Training Loss')
plt.plot(m2.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(m2.history['accuracy'], label='Training accuracy')
plt.plot(m2.history['val_accuracy'], label='Validation accuracy')
plt.title('precision over Epochs')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.tight_layout()
plt.show()

from keras.preprocessing.image import load_img, img_to_array
img = load_img('E://CNN/ship.jpg')
img = img_to_array(img)
img = img.reshape(1,32,32,3) 
img = np.expand_dims(img, 0)
img = img.astype('float32')
img = img/255.0

pred = model.predict(img)
