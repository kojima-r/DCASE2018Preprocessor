import librosa
import scipy
import glob
import numpy as np
from multiprocessing import Pool

def get_feature(filename,feature):
    y, sr = librosa.load(filename)
    if feature=="mfcc":
        mfcc_feature = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc_feature)
        mfcc_deltadelta = librosa.feature.delta(mfcc_delta)
        f=np.vstack([mfcc_feature, mfcc_delta,mfcc_deltadelta])
        return f
    elif feature=="mel":
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        logS = librosa.amplitude_to_db(S, ref=np.max)
        return logS
    elif feature=="mel2":
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        logS = librosa.amplitude_to_db(S, ref=np.max)
        logS_delta = librosa.feature.delta(logS)
        logS_deltadelta = librosa.feature.delta(logS)
        f=np.vstack([logS, logS_delta, logS_deltadelta])
        return f
    elif feature=="spec":
        # win_length =n_fft
        # hop_length=win_length / 4 
        D = librosa.stft(y,n_fft=1024,hop_length=None, win_length=None)
        log_power = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return log_power

def process(args):
    el=args["label"]
    name=args["name"]
    feature=args["feature"]
    filename="wav/"+name+".wav"
    outfilename="npy/"+name+"."+feature+".npy"
    print("[LOAD]",filename)
    feature=get_feature(filename,feature)

    np.save(outfilename,feature)
    l=feature.shape[1]
    return l

def main():
    data=[]
    filename="BirdVox-DCASE-20k.csv"
    #feature="mfcc"
    feature="spec"
    fp=open(filename)
    next(fp)
    for line in fp:
        arr=line.split(",")
        name=arr[0]
        label=arr[2]
        data.append({"label":label,"name":name,"feature":feature})
        #filename="wav/28d7bccb-30ce-40ae-a33a-c0bb3cb4c1e5.wav"
        #f=get_feature(filename)

    p = Pool(64)
    results=p.map(process, data)
    p.close()


    ml=max(results)
    print(ml)

    #print(ml)
if __name__ == '__main__':
    main()
