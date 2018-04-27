from pathlib import Path
import numpy as np
from scipy.signal import resample

## data loading
datapath = Path('../dataset')
xfile = 'X_features_spec.npy'
yfile = 'Y_labels_spec.npy'
def load_waveforms():
    X_list = np.load(str(datapath.joinpath(xfile)))
    Y_list = np.load(str(datapath.joinpath(yfile)))
    return X_list, Y_list

def positve_samples(xlist):
    ## some samples have negative signs
    xl_new = []
    for sample in range(xlist.shape[0]):
        points = xlist[sample]
        for p in range(points.shape[0]):
            point = points[p]
            if np.sum(point) < 0:
                points[p] = -point
        xl_new.append(points)
    return np.array(xl_new)

## apply to loaded dataset
def split_channel(xlist):
    ## this method is out of date since the dimension might be other than 625
    ## input as (n, 2500)
    def standard_resample(arr):
        return resample(arr, 2500)
    ## if some is not with dim 625, resample it
    xl_new = []
    for sample in range(xlist.shape[0]):
        points = xlist[sample]
        if points.shape[1] != 2500:
            print("resample")
            print(points.shape)
            points = np.apply_along_axis(standard_resample, axis=1, arr=points)
        points = points.reshape((points.shape[0], 4, 625))
        xl_new.append(points)
    return np.array(xl_new)

## input is after split
def apply_resample(xlist, outdim):
    ## resample
    def resample_waveform(arr):
        ## arr.shape: (indim, )
        return resample(arr, outdim)
    xl_new = []
    for sample in range(xlist.shape[0]):
        points = xlist[sample]
        points = np.apply_along_axis(resample_waveform, axis=2, arr=points)
        xl_new.append(points)
    return np.array(xl_new)

## input is combined exp. (18000 ,625, 4)
def get_xtrain_mean(x_train):
    ## mean value for each dimension (exp. each of 625 dim)
    m = np.mean(x_train, axis=0)
    ## then we can apply x_train - m for zero mean
    return m

## input is after split
## one variance for each channel
def normalize_waveform():
    ## we don't necessarily need this
    pass

def combine_samples(arrs):
    ## exp. arrs.shape: (20, ?)
    pass

def binary_label(ylist):
    ## 1, 2 --> 1
    ylist_new = []
    for sample in range(ylist.shape[0]):
        labels = ylist[sample]
        labels[labels > 1] = 1
        ylist_new.append(labels)
    return np.array(ylist_new)

def combine_samples(arrs):
    ## exp. arrs.shape: (20, ?)
    if arrs.shape[0] < 1:
        return arrs
    sp = list(arrs[0].shape)
    sp[0] = 0
    combined = np.zeros(sp)
    print("combinde", combined.shape)
    for sample in range(arrs.shape[0]):
        arr = arrs[sample]
        combined = np.concatenate((combined, arr), axis=0)
    return combined