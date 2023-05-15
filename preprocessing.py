import pandas as pd
import librosa as lb
import numpy as np
import os
import math
import json

#function to read in sound files and convert to mfccs along with gathering necessary metadata for use in CNN model
def create_mfccs(dataPath, sampleRate, nMfcc, nFft, hopLength, numSegments):
    
    
    metaData = pd.read_csv('dataset/musicnet_metadata.csv')
    
    #create key-value pair for all unique composers
    composerLabelKey = {}
    composers = np.unique(metaData['composer'])
    for i in range(len(composers)):
        composerLabelKey[composers[i]] = i
        
    #dictionary to organize and store data
    data = {
        'composer': [],
        'mfcc': [],
        'label': []
    }
    
    files = os.listdir(dataPath)
    
    #calculate the number of samples in the shortest track to normalize output size
    minSamplesInTrack = sampleRate * min(metaData['seconds'])
    nSamplesPerSegment = math.floor(minSamplesInTrack/numSegments)
    
    #loop through all the audio files and convert audio data to mfcc
    for file in files:
        
        #track id is the same as the title of the sound file
        id = int(file.strip('.wav'))
        filePath = os.path.join(dataPath, file)
        signal, sr = lb.load(filePath, sr=sampleRate)
        
        #partition mfccs based on the number of segments
        for i in range(numSegments):
            startSample = nSamplesPerSegment * i 
            
            finishSample = startSample + nSamplesPerSegment
            
            mfcc = lb.feature.mfcc(y=signal[startSample:finishSample], sr=sr, n_fft=nFft,
                                   n_mfcc=nMfcc, hop_length=hopLength)
            
            #transpose mfcc
            mfcc = mfcc.T
            
            #get coresponding composer from track id
            composer = metaData[metaData['id'] == id]['composer'].item()
            data['mfcc'].append(mfcc.tolist())
            data['composer'].append(composer)
            data['label'].append(composerLabelKey[composer])

    return data, composerLabelKey

#function to write the data dictionary and composer key-value pair to seperate json files 
def write_to_database(path, data, key):
    with open(path + 'data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    #writing this to a file in case I make an application and want the composers name outputted as well
    with open(path + 'key.json', 'w') as k:
        json.dump(key, k, indent=4)
  
  
if __name__ == "__main__":
    data, key = create_mfccs(dataPath='dataset/musicnet/musicnet/train_data/', sampleRate=22050, 
                           nMfcc=13, nFft=2048, hopLength=512, numSegments=5)
    
    write_to_database(path='preprocessed_data/', data=data, key=key)
    