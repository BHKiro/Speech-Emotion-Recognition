import SER
import numpy as np
import os
from json_tricks import dump,load
import time
import librosa  # For feature extraction
import tensorflow as tf
import keras
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
from datetime import date


# Load X,Y json files back into lists, convert to np.arrays

x_path = './JSON Files/X_datanew_5features.json'  # FILE LOAD PATH
X = load(x_path)
X = np.asarray(X, dtype='float32')

y_path = './JSON Files/Y_datanew.json'  # FILE LOAD PATH
Y = load(y_path)
Y = np.asarray(Y, dtype='int8')

# Split to train, validation, and test sets.
x_train, x_tosplit, y_train, y_tosplit = train_test_split(X, Y, test_size=0.125, random_state=1)

x_val, x_test, y_val, y_test = train_test_split(x_tosplit, y_tosplit, test_size=0.304, random_state=1)

# vectors for Y: emotion classification
y_train_class = tf.keras.utils.to_categorical(y_train, 8, dtype='int8')
y_val_class = tf.keras.utils.to_categorical(y_val, 8, dtype='int8')

# Save x_test, y_test to JSON.
file_path = 'x_test_data.json'
dump(obj=x_test, fp=file_path)

file_path = 'y_test_data.json'
dump(obj=y_test, fp=file_path)

# Machine learning model
model = Sequential()
model.add(layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1:3])))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(64))
model.add(layers.Dense(8, activation='softmax'))
print(model.summary())

batch_size = 13

# Callbacks functions
checkpoint_path = './best_weights.hdf5'

# -> Save the best weights
mcp_save = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                     monitor='val_categorical_accuracy',
                                     mode='max')

# -> Reduce learning rate after 100 epoches without improvement.
rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                    factor=0.1, patience=100)

# Compile & train
model.compile(loss='categorical_crossentropy',
              optimizer='RMSProp',
              metrics=['categorical_accuracy'])

model.save("SER_Model")

history = model.fit(x_train, y_train_class,
                    epochs=400, batch_size=batch_size,
                    validation_data=(x_val, y_val_class),
                    callbacks=[mcp_save, rlrop])
# Define the best weights to the model.
model.load_weights(checkpoint_path)

date = date.today()

y_val_data = y_val_class.tolist()
y_val_path = f'./JSON Files/Y_validation_data_test.json'  # Validation set SAVE PATH{date}
dump(obj=y_val_data, fp=y_val_path)

y_pred_data = model.predict(x_val).tolist()
y_pred_path = f'./JSON Files/Y_prediction_data_test.json'  # Predictions SAVE PATH{date}
dump(obj=y_pred_data, fp=y_pred_path)

model_path = '/content/drive/My Drive/Colab Notebooks/model0805.json'
model_json = model.to_json()
dump(obj=model_json, fp=model_path)

# Loss, Accuracy presentation

# Plot history: Loss
plt.figure(1)
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss for train and validation')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")

# Plot history: Accuracy
plt.figure(2)
plt.plot(history.history['categorical_accuracy'], label='Acc (training data)')
plt.plot(history.history['val_categorical_accuracy'], label='Acc (validation data)')
plt.title('Model accuracy')
plt.ylabel('Acc %')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
