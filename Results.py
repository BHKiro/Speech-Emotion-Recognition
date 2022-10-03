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
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np


y_val_path = './JSON Files/Y_validation_data_test.json'  # FILE LOAD PATH
y_val_class = load(y_val_path)
y_val_class = np.asarray(y_val_class, dtype='float32')

y_val_path = './JSON Files/Y_prediction_data_test.json'  # FILE LOAD PATH
predictions = load(y_val_path)
predictions = np.asarray(predictions, dtype='float32')

# Validation Confusion matrix
y_val_class = np.argmax(y_val_class, axis=1)
y_pred_class = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_val_class, y_pred_class) # normalize='true'

index = ['calm', 'happy', 'sad', 'angry']  # ['neutral', 'fearful', 'disgust', 'surprised']
columns = ['calm', 'happy', 'sad',
           'angry']  # ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

            # 01 - neutral

            # 02 - calm
            # 03 - happy
            # 04 - sad
            # 05 = angry

            # 06 = fearful
            # 07 = disgust
            # 08 = surprised
            ##

cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(12, 8))
ax = plt.axes()

sns.heatmap(cm_df, ax=ax, cmap='PuBu', annot=True)
ax.set_ylabel('True emotion')
ax.set_xlabel('Predicted emotion')

# Validation set prediction accuracy rates

cm = confusion_matrix(y_val_class, y_pred_class)
values = cm.diagonal()
row_sum = np.sum(cm,axis=1)
acc = values / row_sum

print('Validation set predicted emotions accuracy:')
for e in range(0, len(values)):
    print(index[e],':', f"{(acc[e]):0.4f}")
