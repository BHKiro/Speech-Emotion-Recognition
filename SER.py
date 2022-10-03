import os

import librosa
import noisereduce as nr
import numpy as np
import time
from pydub import AudioSegment, effects


def emotion(emotionNum):
    """This Function reads in the string that contains a number (1-7) and returns a number (0-7)."""
    if emotionNum == "01":
        return 0
    elif emotionNum == "02":
        return 1
    elif emotionNum == "03":
        return 2
    elif emotionNum == "04":
        return 3
    elif emotionNum == "05":
        return 4
    elif emotionNum == "06":
        return 5
    elif emotionNum == "07":
        return 6
    else:
        return 7


def genderID(genderNum):
    """This function determines whether the speaker is a male or a female | Inputs - genderNum: number | Outputs: 1(male)
    or 2 (female)"""
    # As the number in the filename specifies the gender in a specific way (Odd:Male,Even:Female) this function will
    # translate those numbers into 1 and 2
    if (int(genderNum) % 2) == 0:
        return 1
    else:
        return 2


def maxLength(folderPath):
    """This function will determine the longest audio signal out of the folder specified | Inputs - FolderPath: a
    path to the folder containing audio files | Outputs - A number that represents sample count in the longest audio
    file in the folder """
    sampleLengths = []
    for subdir, dirs, files in os.walk(folderPath):
        for file in files:
            # Importing the audio signal as an array
            x, fs = librosa.load(path=os.path.join(subdir, file),
                                 sr=None)  # the sample rate is used for librosa's MFCCs.

            sampleLengths.append(len(x))
    max_sample = np.max(sampleLengths)
    print('Maximum sample length: ', max_sample)
    return max_sample


def feature_extraction(folderPath, totalLength, frameLength, hopLength):
    """"This function will extract features from the audio files in the specified folder (database)| :param
    folderPath: a path to the folder (database), totalLength: the number of samples in the longest audio file,
    frameLength: length of a frame (for some MIR functions) in samples (if in doubt insert 2048), hopLength: length
    of a hop (for some MIR functions) in samples (if in doubt insert 512) | :return: - rms, zerocross, mfcc, emotions,
    spectral_centroid, gender as arrays """
    # Creating empty arrays to store features in
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

            # Filling the data lists
            rms.append(f1)
            zerocross.append(f2)
            mfcc.append(f3)
            spectral_centroid.append(f4)
            emotions.append(emotion(name))
            gender.append(genderID(genderNum))

    return rms, zerocross, mfcc, emotions, spectral_centroid, gender


def demo_feature(path_to_file, frameLength=2048, hopLength=512, gender=1):
    """" This function will read in an audio file and extract features from it (was used for testing of the accuracy
    of the algorithm) """
    # Fetch sample rate.
    _, fs = librosa.load(path=path_to_file, sr=None)
    # Load audio file
    rawsound = AudioSegment.from_file(path_to_file, duration=None)
    # Normalize to 5 dBFS
    normalizedsound = effects.normalize(rawsound, headroom=5.0)
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
    # Noise reduction
    final_audio = nr.reduce_noise(normal_x, sr=fs)

    # Features extraction
    f1 = librosa.feature.rms(final_audio, frame_length=frameLength, hop_length=hopLength).T
    print('Energy shape:', f1.shape)

    f2 = librosa.feature.zero_crossing_rate(final_audio, frame_length=frameLength, hop_length=hopLength,center=True).T  # ZCR
    print('ZCR shape:', f2.shape)

    f3 = librosa.feature.mfcc(final_audio, sr=fs, S=None, n_mfcc=13, hop_length = hopLength).T  # MFCC
    print('MFCCs shape:', f3.shape)

    f4 = librosa.feature.spectral_centroid(final_audio, sr=fs, S=None, n_fft=2048, hop_length=hopLength).T
    print('SC shape:', f4.shape)

    gender = np.asarray(gender).astype('int8')

    gender = np.resize(gender, (495, 13)) #339 #597
    f1 = np.resize(f1, (495, 1))
    f2 = np.resize(f1, (495, 1))
    f3 = np.resize(f1, (495, 13))
    f4 = np.resize(f1, (495, 1))

    X = np.concatenate((f1, f2, f3, f4, gender), axis=1)

    X_out_3D = np.expand_dims(X, axis=0)
    return X_out_3D


def is_silent(data):
    """:return: 'True' if below the 'silent' threshold"""
    return max(data) < 100

def TicTocGenerator():
    """
    :return: time difference
    """
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    """
    Prints the time difference yielded by generator instance TicToc
    """
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    """
    Records a time in TicToc, marks the beginning of a time interval
    """
    toc(False)
