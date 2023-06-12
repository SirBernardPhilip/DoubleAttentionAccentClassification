import pickle
import numpy as np
from random import randint, randrange
from torch.utils import data
import math
import librosa
import warnings
from featureExtractor import mfsc
warnings.filterwarnings("ignore")



import random
import numpy as np



def featureReader(featurePath, VAD=None):

    with open(featurePath,'rb') as pickleFile:
        features = pickle.load(pickleFile)
        if VAD is not None:
            filtered_features = VAD.filter(features)
        else:
            filtered_features = features

    if filtered_features.shape[1]>0.:
        return np.transpose(filtered_features)
    else:
        return np.transpose(features)

def normalizeFeatures(features, normalization='cmn'):

    mean = np.mean(features, axis=0)
    features -= mean 
    if normalization=='cmn':
       return features
    if normalization=='cmvn':
        std = np.std(features, axis=0)
        std = np.where(std>0.01,std,1.0)
        return features/std

class Dataset(data.Dataset):

    def __init__(self, utterances, parameters, augment, crop = True, padding = False, old = False, num_classes = 5):
        'Initialization'
        self.utterances = utterances
        self.parameters = parameters
        self.num_samples = len(utterances)
        self.crop = crop
        self.padding = padding
        self.old = old
        self.num_classes = num_classes
        self.augment = augment

    def __normalize(self, features):
        mean = np.mean(features, axis=0)
        features -= mean 
        if self.parameters.normalization=='cmn':
            return features
        if self.parameters.normalization=='cmvn':
            std = np.std(features, axis=0)
            std = np.where(std>0.01,std,1.0)
            return features/std

    def __sampleSpectogramWindow(self, features):
        if self.crop:
            file_size = features.shape[0]
            windowSizeInFrames = self.parameters.window_size*100
            index = randint(0, max(0,file_size-windowSizeInFrames-1))
            a = np.array(range(min(file_size, int(windowSizeInFrames))))+index
            sliced_spectrogram = features[a,:]
            # Add zero padding if the spectrogram is shorter than the fixed length
            if self.padding and (sliced_spectrogram.shape[0] < windowSizeInFrames):
                padding = np.zeros((int(windowSizeInFrames) - sliced_spectrogram.shape[0], sliced_spectrogram.shape[1]), dtype=np.float32)
                sliced_spectrogram = np.concatenate((sliced_spectrogram, padding), axis=0)
            return sliced_spectrogram
        else:
            return features

    def __getFeatureVector(self, utteranceName):
        if (self.old):
            with open(utteranceName.replace('features', 'features_old'),'rb') as pickleFile:
                features = pickle.load(pickleFile)
            windowedFeatures = self.__normalize(np.transpose(features))
            return windowedFeatures            
     
        with open(utteranceName,'rb') as pickleFile:
            features = pickle.load(pickleFile)
        windowedFeatures = self.__sampleSpectogramWindow(self.__normalize(np.transpose(features["features"])))
        return windowedFeatures            
     
    def __len__(self):
        return self.num_samples
    
    def __augment(self, utteranceName):
        # With 0.25 probability, we return as is
        if (randrange(0, 4) == 0):
            return self.__getFeatureVector(utteranceName)
        path = utteranceName.replace("pickle", "wav").replace("features_old", "clips_wav").replace("features", "clips_wav")
        y, sfreq = librosa.load(path, sr=None)
        # pick noise, pitch shifting noise or time stretching
        option = randrange(0, 3)
        if (option == 0):
            # pitch shifting
            shift = randrange(-4, 4)
            y = librosa.effects.pitch_shift(y, sr=sfreq, bins_per_octave=16, n_steps=shift)
        elif (option == 1):
            # time stretching
            stretch = randrange(7, 13)/10
            y = librosa.effects.time_stretch(y, rate=stretch)
        else:
            # noise
            noise = np.random.randn(len(y))
            y = y + 0.005*noise
        mf = mfsc(y, sfreq)
        return self.__sampleSpectogramWindow(self.__normalize(np.transpose(mf)))

    def __getitem__(self, index):
        'Generates one sample of data'
        utteranceTuple = self.utterances[index].strip().split()
        utteranceName = utteranceTuple[0].replace("cv11.0", "cv11")
        utteranceNumber = int(utteranceTuple[1])
        if (self.num_classes == 4):
            if utteranceNumber == 4:
                utteranceNumber = 0
        elif (self.num_classes == 3):
            if (utteranceNumber == 4) or (utteranceNumber == 3):
                utteranceNumber = 0
        elif (self.num_classes == 2):
            if (utteranceNumber == 4) or (utteranceNumber == 3):
                utteranceNumber = 0
            elif (utteranceNumber == 2):
                utteranceNumber = 1
        utteranceLabel =np.array(utteranceNumber)
        if self.augment:
            features = self.__augment(utteranceName)
        else:
            features = self.__getFeatureVector(utteranceName)
        return features, utteranceLabel
