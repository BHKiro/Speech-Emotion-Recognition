import os
from json_tricks import load
import numpy as np
import librosa
from pydub import AudioSegment, effects
import noisereduce as nr
import tensorflow as tf
import keras
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import SER

# Declaring paths to the model, the weights and the .wav file recording of your choice
# Path to audio recording
recording_path = './my_recording_5.wav'

# Path to the model trained with all 8 emotions
saved_model_path = 'PATH_TO_MODEL'
saved_weights_path = 'PATH_TO_WEIGHTS'

# Loading the model and the weights
model = tf.keras.models.load_model(saved_model_path)
model.load_weights(saved_weights_path)

# Compiling the model with similar specification as the original model.
model.compile(loss='categorical_crossentropy',
              optimizer='RMSProp',
              metrics=['categorical_accuracy'])
# Printing model summary
print(model.summary())

# Emotions list is created for a readable form of the model prediction.
emotions = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'suprised'
}
emo_list = list(emotions.values())
total_predictions = []  # A list for all predictions in the session.

X = SER.demo_feature(recording_path)

predictions = model.predict(X, use_multiprocessing=True)
pred_list = list(predictions)

pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0)  # Get rid of 'array' & 'dtype' statments.
total_predictions.append(pred_np)

# Present emotion distribution for a sequence
fig = plt.figure(figsize=(10, 2))
plt.bar(emo_list, pred_np, color='green')
plt.ylabel("Probabilty (%)")
plt.show()

max_emo = np.argmax(predictions)
print('max emotion:', emotions.get(max_emo, -1))
print(100 * '-')
