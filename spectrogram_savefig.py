# imports
from pathlib import Path
import numpy as np
from scipy.signal import resample
from scipy.signal import spectrogram
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, BatchNormalization, MaxPooling1D
from keras import regularizers
from keras import initializers
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

## utitlities
## data loading
datapath = Path('../dataset')
xfile = 'X_features_spec_aligned.npy'
yfile = 'Y_labels_spec.npy'

def load_waveforms():
    X_list = np.load(str(datapath.joinpath(xfile)))
    Y_list = np.load(str(datapath.joinpath(yfile)))
    return X_list, Y_list


x_list, y_list = load_waveforms()
for i in range(x_list.shape[0]):
    print(x_list[i].shape)

## short time fourier transformation with window and overlapping
## 1000 per channel
win_size = 128
win_overlap = 64
win_size = input("Please set the window size for the spectrogram:")
win_size = int(win_size)
win_overlap = input("Please set the window overlap for the spectrogram:")
win_overlap = int(win_overlap)

print("Spectrogram has window size of {} and overlap of {}".format(win_size, win_overlap))

def transform_to_spectrogram(arr):
    sp = spectrogram(arr, window='hanning', nperseg=win_size, noverlap=win_overlap)
    return sp[2]

x_specgrams = []
for sample in range(x_list.shape[0]):
    points = x_list[sample]
    points = np.apply_along_axis(transform_to_spectrogram, axis=2, arr=points)
    print(points.shape)
    x_specgrams.append(points)

## experiment: save to folder and files
cid_file = '../dataset/cid.npy'
n_coord_file = '../dataset/n_coords_aligned.npy'
cid = np.load(cid_file)
n_coord = np.load(n_coord_file)
specp = Path('./reports/spectrogram/windowSize_{}_overlap_{}'.format(win_size, win_overlap))
print('Save to ', specp)
if specp.exists() == False:
    specp.mkdir(parents=True)

start_id = 0
start_id = input("Start from case ? (0 ~ 20):")
start_id = int(start_id)

for case_id in range(start_id, x_list.shape[0]):
    case_name = cid[case_id]
    print("case name", case_name)
#     cur_p = specp.joinpath('{}'.format(case_id))
    cur_p = specp.joinpath('{}_{}'.format(case_name, n_coord[case_id].shape[0]))
    if cur_p.exists() == False:
        cur_p.mkdir(parents=True)
        
    fxs = x_specgrams[case_id][:][:][:, :].flatten()

    bin_num = 100

    fx0 = x_specgrams[case_id][:,0].flatten()
    fx1 = x_specgrams[case_id][:,1].flatten()
    fx2 = x_specgrams[case_id][:,2].flatten()
    fx3 = x_specgrams[case_id][:,3].flatten()

    amin = fxs.min()
    amax = np.min([fx0.max(), fx1.max(), fx2.max(), fx3.max()])
    xlabels = np.arange(amin, amax, (amax-amin)/bin_num)
    xlabels = xlabels[0:100]
    hist1 = np.histogram(fx0, range=(amin, amax), bins=bin_num) ## histogram of amplitude by number
    hist2 = np.histogram(fx1, range=(amin, amax), bins=bin_num) ## histogram of amplitude by number
    hist3 = np.histogram(fx2, range=(amin, amax), bins=bin_num) ## histogram of amplitude by number
    hist4 = np.histogram(fx3, range=(amin, amax), bins=bin_num) ## histogram of amplitude by number

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, sharey=True)
    ax1.plot(xlabels, hist1[0])
    # ax1.set_xticks(xlabels)
    ax1.set_title('histogram of amplitude count (how amplitude of frequencies are distributed)\n channel 1')
    ax1.set_ylabel("point count")
    ax1.set_xlabel("amplitude")
    ax2.plot(xlabels, hist2[0])
    ax2.set_title('channel 2')
    ax3.plot(xlabels, hist3[0])
    ax3.set_title('channel 3')
    ax4.plot(xlabels, hist4[0])
    ax4.set_title('channel 4')
    f.subplots_adjust(hspace=0.3)
    fig = plt.gcf()
    fig.set_size_inches(12.5, 16.5)
    plt.savefig(str(cur_p.joinpath('amplitude count.png')))
    plt.clf()
    
    all_sp = np.transpose(x_specgrams[case_id], axes=[1,2,0,3])

    all_sp_1 = all_sp[0].reshape(-1, all_sp[0].shape[-1] * all_sp[0].shape[-2])
    mean_spec_1 = np.apply_along_axis(np.mean, axis=1, arr=all_sp_1) ## histogram of frequency / mean amplitude
    all_sp_2 = all_sp[1].reshape(-1, all_sp[1].shape[-1] * all_sp[1].shape[-2])
    mean_spec_2 = np.apply_along_axis(np.mean, axis=1, arr=all_sp_2) ## histogram of frequency / mean amplitude
    all_sp_3 = all_sp[2].reshape(-1, all_sp[2].shape[-1] * all_sp[2].shape[-2])
    mean_spec_3 = np.apply_along_axis(np.mean, axis=1, arr=all_sp_3) ## histogram of frequency / mean amplitude
    all_sp_4 = all_sp[3].reshape(-1, all_sp[3].shape[-1] * all_sp[3].shape[-2])
    mean_spec_4 = np.apply_along_axis(np.mean, axis=1, arr=all_sp_4) ## histogram of frequency / mean amplitude
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    ax1.plot(mean_spec_1)
    ax1.set_title('mean spectrum of a window (of all points) \nchannel 1')
    ax1.set_ylabel("amplitude")
    ax1.set_xlabel("frequency")
    ax2.plot(mean_spec_2)
    ax2.set_title('channel 2')
    ax3.plot(mean_spec_3)
    ax3.set_title('channel 3')
    ax4.plot(mean_spec_4)
    ax4.set_title('channel 4')
    fig = plt.gcf()
    fig.set_size_inches(10.5, 15.5)
    plt.savefig(str(cur_p.joinpath('mean sepectrum.png')))
    plt.clf()
    
    sp1 = x_specgrams[case_id][:,0,:]
    sp2 = x_specgrams[case_id][:,1,:]
    sp3 = x_specgrams[case_id][:,2,:]
    sp4 = x_specgrams[case_id][:,3,:]
    point_lim = 100
    ############
    for p_i in range(sp1.shape[0]):
        print("point ", p_i)
        if p_i >= point_lim:
            break
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, sharey=True)
        p_coord = n_coord[case_id][p_i]
        label = y_list[case_id][p_i]
        sg1 = sp1[p_i]
        sg2 = sp2[p_i]
        sg3 = sp3[p_i]
        sg4 = sp4[p_i]
        sns.heatmap(sg1, ax=ax1)
        sns.heatmap(sg2, ax=ax2)
        sns.heatmap(sg3, ax=ax3)
        sns.heatmap(sg4, ax=ax4)
        ax1.set_title('spectrograms\n channel 1')
        ax1.set_ylabel("frequency")
        ax1.set_xlabel("time window")
        ax2.set_ylabel("frequency")
        ax2.set_xlabel("time window")
        ax3.set_ylabel("frequency")
        ax3.set_xlabel("time window")
        ax4.set_ylabel("frequency")
        ax4.set_xlabel("time window")
        fig = plt.gcf()
        fig.set_size_inches(10.5, 18.5)
        plt.savefig(str(cur_p.joinpath('spectrogram_{}_{}_{}.png'.format(label, p_coord[0], p_coord[1]))))
        plt.clf()
