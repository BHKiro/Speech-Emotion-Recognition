from SER import genderID, emotionfix
from json import dump
import librosa
import numpy as np
import noisereduce as nr
import os

def feature_extraction(folderPath, totalLength, frameLength, hopLength):
    rms = []
    mfcc = []
    zerocross = []
    emotions = []
    spectral_centroid = []
    gender = []

    for subdir, dirs, files in os.walk(folderPath):
        for file in files:
            # Importing the audio signal as an array
            x, fs = librosa.load(path=os.path.join(subdir, file),
                                 sr=None)  # the sample rate is used for librosa's MFCCs.

            # Normalizing the signal
            normal_x = x / max(abs(x))

            # Trim silence from the beginning and the end.
            xt, index = librosa.effects.trim(normal_x, top_db=30, frame_length=2048, hop_length=512)

            # Pad for duration equalization.
            padded_x = np.pad(xt, (0, totalLength - len(xt)), mode='constant')

            # Noise reduction.
            final_x = nr.reduce_noise(padded_x, sr=fs)

            # Features extraction
            f1 = librosa.feature.rms(final_x, frame_length=frameLength,
                                     hop_length=hopLength)  # Energy - Root Mean Square
            # print('Energy shape:', f1.shape)

            f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frameLength, hop_length=hopLength,
                                                    center=True)  # ZCR
            # print('ZCR shape:', f2.shape)

            f3 = librosa.feature.mfcc(final_x, sr=fs, S=None, n_mfcc=13, hop_length=hopLength)  # MFCC
            # print('MFCCs shape:', f3.shape)

            f4 = librosa.feature.spectral_centroid(final_x, sr=fs, S=None, n_fft=2048, hop_length=hopLength)

            # Getting the name from RAVDESS files
            name = file[6:8]

            # Gender
            genderNum = file[18:20]

            # 1 - neutral
            # 2 - calm
            # 3 - happy
            # 4 - sad
            # 05 = angry
            # 06 = fearful
            # 07 = disgust
            # 08 = surprised
            ##
            if name == "02" or name == "03" or name == "04" or name == "05":
                # Filling the data lists
                rms.append(f1)
                zerocross.append(f2)
                mfcc.append(f3)
                spectral_centroid.append(f4)
                emotions.append(emotionfix(name))
                gender.append(genderID(genderNum))

    return rms, zerocross, mfcc, emotions, spectral_centroid, gender

DATASERT_PATH = '/content/drive/My Drive/Colab Notebooks/RAVDESS'
rms, zerocross, mfcc, emotions, spectral_centroid, gender = feature_extraction(folderPath=DATASERT_PATH, totalLength=495, frameLength=2048, hopLength=512)

# Root mean square
f_rms = np.asarray(rms).astype('float32')
f_rms = np.swapaxes(f_rms, 1, 2)
print('RMS shape:', f_rms.shape)

# Zero Crossing Rate
f_zerocross = np.asarray(zerocross).astype('float32')
f_zerocross = np.swapaxes(f_zerocross, 1, 2)
print('ZCR shape:', f_zerocross.shape)

# MFCC
f_mfccs = np.asarray(mfcc).astype('float32')
f_mfccs = np.swapaxes(f_mfccs, 1, 2)
print('MFCC shape:', f_mfccs.shape)

# CentralSpectroid
f_spectral_centroid = np.asarray(spectral_centroid).astype('float32')
f_spectral_centroid = np.swapaxes(f_spectral_centroid,1,2)
print('Spec Centroid shape:', f_mfccs.shape)

# Gender
f_gender = np.asarray(gender).astype('int8')
f_gender = np.resize(f_gender, (768, 495, 13))
print('Gender Array', f_gender.shape)

# Select what features will be included in the testing


# Concatenating all features to 'X' variable.
X = np.concatenate((f_zerocross, f_rms, f_mfccs, f_spectral_centroid), axis=2)
Z = np.concatenate((f_zerocross, f_rms, f_mfccs), axis=2)
K = np.concatenate((f_gender, f_zerocross, f_rms, f_mfccs, f_spectral_centroid), axis=2)

# Preparing 'Y' as a 2D shaped variable.
Y = np.asarray(emotions).astype('int8')
Y = np.expand_dims(Y, axis=1)
print(Y.shape)

# Save X,Y arrays as lists to json files.
k_data = K.tolist()
k_path = '/content/drive/My Drive/Colab Notebooks/X_datanew_5features.json'  # FILE SAVE PATH
dump(obj=k_data, fp=k_path)

x_data = X.tolist()
x_path = '/content/drive/My Drive/Colab Notebooks/X_datanew_4features.json'  # FILE SAVE PATH
dump(obj=x_data, fp=x_path)

z_data = Z.tolist()
z_path = '/content/drive/My Drive/Colab Notebooks/X_datanew_3features.json'  # FILE SAVE PATH
dump(obj=z_data, fp=z_path)

y_data = Y.tolist()
y_path = '/content/drive/My Drive/Colab Notebooks/Y_datanew.json'  # FILE SAVE PATH
dump(obj=y_data, fp=y_path)
