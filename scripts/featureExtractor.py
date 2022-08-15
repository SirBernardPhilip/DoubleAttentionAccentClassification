import numpy as np
import soundfile as sf
import sys
import pickle
import librosa
import argparse

# In this particular case we ignore warnings of loading a .m4a audio
# Not a good practice
import warnings
warnings.filterwarnings("ignore")

def mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', n_mels=80, preemCoef=0.97):
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 0
    highfreq = sfr/2

    # melspectrogram
    y *= 32768
    y[1:] = y[1:] - preemCoef*y[:-1]
    y[0] *= (1 - preemCoef)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
    D = np.abs(S)
    param = librosa.feature.melspectrogram(S=D, sr=sfr, n_mels=n_mels, fmin=lowfreq, fmax=highfreq, norm=None)
    mf = np.log(np.maximum(1, param))
    return mf

def normalize(features):
    return features-np.mean(features, axis=0)


def extractFeatures(audioPath):

    y, sfreq = sf.read(audioPath)
    features = mfsc(y, sfreq)    
    return normalize(np.transpose(features))

def main(params):

    # Doing this to be able to print progress of processing files
    with open(params.audioFilesList,'r') as filesFile:
        total_lines = sum(1 for line in list(filesFile))
        filesFile.close()
    
    with open(params.audioFilesList,'r') as filesFile:
        
        print(f"[Feature Extractor] {total_lines} audios ready for feature extraction.")

        line_num = 0
        for featureFile in filesFile:

            featureFile = featureFile.replace("\n", "")

            # print(f"[Feature Extractor] Processing file {featureFile}...")

            # y, sfreq = sf.read('{}'.format(featureFile[:-1]))
            # y, sfreq = sf.read('{}'.format(featureFile))
            y, sfreq = librosa.load(
                f'{featureFile}',
                sr = 16000,
                mono = True,
                ) 

            mf = mfsc(y, sfreq)

            with open(f'{featureFile[:-4]}.pickle', 'wb') as handle:
                pickle.dump(mf,handle)

            # print(f"[Feature Extractor] File processed. Dumped pickle in {featureFile[:-4]}.pickle")
            
            progress_pctg = line_num / total_lines * 100
            print(f"[Feature Extractor] {progress_pctg:.2f}% audios processed...")
            line_num = line_num + 1

        print(f"[Feature Extractor] All audios processed!")

if __name__=='__main__':

    print(f"[Feature Extractor] Starting audio processing...")
    
    parser = argparse.ArgumentParser(description='Extract Features. Looks for .wav files and extract Features')
    parser.add_argument('--audioFilesList', '-i', type=str, required=True, default='', help='Wav Files List.')
    params = parser.parse_args()

    main(params)