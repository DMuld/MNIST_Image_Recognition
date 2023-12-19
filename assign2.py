from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
# from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

##################PART 1##################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Checking the shape
print("train shape:", x_train.shape, y_train.shape)
print("test shape:", x_test.shape, y_test.shape)

# Rescale the data from 0 to 255 -> 0 to 1 NORMALIZE
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make the graph.
# Shows 16 images
sbplt, img = plt.subplots(1, 16, figsize=(20,20))
for i in range(16):
    sample = x_train[random.randint(0,9)]
    img[i].imshow(sample, cmap='gray')
    img[i].set_title("Count: {}".format(i), fontsize=10)
plt.show()

##################PART 2##################

# Creates the model functionally following the LeNet-5
inputs = keras.Input(shape=(28,28,1))
d1 = keras.layers.Conv2D(6, kernel_size=(5,5), strides=1, activation='tanh', padding='same')(inputs)
d2 = keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(d1)
d3 = keras.layers.Conv2D(16, kernel_size=(5,5), strides=1, activation='tanh', padding='valid')(d2)
d4 = keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(d3)
d5 = keras.layers.Conv2D(120, kernel_size=(5,5), strides=1, activation='tanh', padding='valid')(d4)
d55 = keras.layers.Flatten()(d5)   #Had to flatten this because I was getting errors with mismatch shapes.
d6 = keras.layers.Dense(84, activation='tanh')(d55)
outputs = keras.layers.Dense(10, activation='softmax')(d6)
model = keras.Model(inputs=inputs, outputs=outputs)

# Shows a summary of what the model will look lik 
model.summary()

##################PART 3##################

# Compiles the training alg. ##https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible##
model.compile(optimizer='SGD', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy']) 

# Fits the training model
fittedModel = model.fit(x_train, y_train, batch_size=128, epochs=55, validation_data=(x_test, y_test))
plt.plot(fittedModel.history['loss'])
plt.plot(fittedModel.history['val_loss'])
plt.title("Epoch vs Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Evaluating the model with the test set.
print("EVALUATION:")
loss_val = model.evaluate(x_test, y_test)
print("Loss Value: ", loss_val)



# # # Use Trained Model to Predict the Testing Set
# y_pred = model.predict(x_test)
# for i in range(len(x_test)):
#     print("X-Value: "+str(x_test[i])+", Predicted Y-Value: "+str(y_pred[i][0])+". Actual Y-Value: "+str(y_test[i]))
 
# # Generate the Plot
# plt.plot(x_test, y_test)
# plt.plot(x_test, y_pred)
# plt.show()