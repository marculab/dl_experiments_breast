from pathlib import Path
import numpy as np
from scipy.signal import resample
from keras.callbacks import Callback
import time
import json

## data loading
logpath = Path('../logs')
datapath = Path('../dataset')
xfile = 'X_features_spec.npy'
yfile = 'Y_labels_spec.npy'
xfile_aligned = 'X_features_spec_aligned.npy'

def load_aligned_waveforms():
    X_list = np.load(str(datapath.joinpath(xfile_aligned)))
    Y_list = np.load(str(datapath.joinpath(yfile)))
    return X_list, Y_list

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

def split_four_channels(xlist):
    xl_new = []
    for sample in range(xlist.shape[0]):
        points = xlist[sample]
        print((points.shape[0], 4, points.shape[1]/4))
        points = points.reshape((points.shape[0], 4, int(points.shape[1]/4)))
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

class LogDir():
    def __init__(self, this_log_dir=None):
        self.time_str = time.strftime('%Y %m.%d %H:%M', time.localtime())
        self.this_log_dir = this_log_dir
        self.this_run_dir = None
        self.epoch_dir = {}

        self.Y_TRAIN_FILE = 'y_train.npy'
        self.Y_VAL_FILE = 'y_val.npy'
        self.TRAIN_PREDICT_SCORE_FILE = 'train_predict_score.npy'
        self.VAL_PREDICT_SCORE_FILE = 'val_predict_score.npy'
        self.ORIG_CSV_LOG_FILE = 'csv_log.csv'

    # def make_crossval_logdirs(self, name=None):
    #     if name == None:
    #         name = self.time_str
    #     else:
    #         name = name + ' ' + self.time_str
    #     this_log_dir = logpath.joinpath(name)
    #     if not this_log_dir.exists():
    #         this_log_dir.mkdir(parents=True)
    #     self.this_log_dir = this_log_dir

    def make_crossval_logdirs(self, name):
        this_log_dir = logpath.joinpath(name)
        if not this_log_dir.exists():
            this_log_dir.mkdir(parents=True)
        self.this_log_dir = this_log_dir

    def make_run_dir(self, val_set):
        this_run_dir = self.this_log_dir.joinpath('__' + str(val_set))
        if not this_run_dir.exists():
            this_run_dir.mkdir(parents=True)
        self.this_run_dir = this_run_dir

    def make_epoch_dir(self, epoch_num):
        this_epoch_dir = self.this_run_dir.joinpath('__' + str(epoch_num))
        if not this_epoch_dir.exists():
            this_epoch_dir.mkdir(parents=True)
        self.epoch_dir[epoch_num] = this_epoch_dir

    def save_y_train(self, y_train):
        np.save(str(self.this_run_dir.joinpath(self.Y_TRAIN_FILE)), y_train)
    
    def save_y_val(self, y_val):
        np.save(str(self.this_run_dir.joinpath(self.Y_VAL_FILE)), y_val)

    def save_train_predict_score(self, epoch_num, train_predict_score):
        np.save(str(self.epoch_dir[epoch_num].joinpath(self.TRAIN_PREDICT_SCORE_FILE)), train_predict_score)

    def save_val_predict_score(self, epoch_num, val_predict_score):
        np.save(str(self.epoch_dir[epoch_num].joinpath(self.VAL_PREDICT_SCORE_FILE)), val_predict_score)

    def rm_csv_log(self):
        csv_path = self.this_run_dir.joinpath(self.ORIG_CSV_LOG_FILE)
        if csv_path.exists():
            csv_path.unlink()

    def write_csv_log(self, epoch_num, logs):
        csv_path = self.this_run_dir.joinpath(self.ORIG_CSV_LOG_FILE)
        if ( not csv_path.exists() ):
            ## create new
            with open(str(csv_path), 'a') as outfile:
                outfile.writelines(['epoch,acc,loss,val_acc,val_loss\n', ','.join([str(epoch_num), str(logs['acc']), str(logs['loss']), str(logs['val_acc']), str(logs['val_loss']) + '\n' ])])
                outfile.close()
        else:
            with open(str(csv_path), 'a') as outfile:
                outfile.write(','.join([str(epoch_num), str(logs['acc']), str(logs['loss']), str(logs['val_acc']), str(logs['val_loss']) ]) + '\n')
                outfile.close()

class LogMeasure(Callback):
    def __init__(self, model, val_set, train_x, train_y, val_x, val_y, name):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.model = model ## reference to the model
        self.val_set = val_set
        self.logdir = LogDir()

        self.logdir.make_crossval_logdirs(name=name)
        self.logdir.make_run_dir(val_set)
        self.logdir.rm_csv_log()

    def on_train_begin(self, logs={}):
        self.losses = []
        self.logdir.save_y_train(self.train_y)
        self.logdir.save_y_val(self.val_y)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        ## save the predictions of the current model to logs folder
        self.logdir.make_epoch_dir(epoch)

        val_predict_score = self.model.predict(self.val_x)
        self.logdir.save_val_predict_score(epoch, val_predict_score)

        train_predict_score = self.model.predict(self.train_x)
        self.logdir.save_train_predict_score(epoch, train_predict_score)

        self.logdir.write_csv_log(epoch, logs)

        # ## use 0 as true
        # pred_score = result[:, 0]
        # fpr, tpr, thresholds = roc_curve(val_list_y[0:1000], pred_score, pos_label=0)
        # count = 10
        # print("thresholds", thresholds[np.arange(0, thresholds.shape[0], int(thresholds.shape[0]/count))])
        # # plt.plot(fpr, tpr)
        
        # new_thres = thresholds[int(thresholds.shape[0] / 2)]
        # y_pred = result[:, 0] < new_thres
        # # y_pred = result
        # # y_pred = np.apply_along_axis(np.argmax, axis=1, arr=y_pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(val_list_y[0:1000], y_pred, average=None)
        # print("precision", precision)
        # print("recall", recall)
        # print("f_score", f_score)
        # print("support", support)
        
        # print("What about train?")
        # result_train = model.predict(train_list_x[0:3000])
        # y_pred_train = result_train[:, 0] < new_thres
        # precision, recall, f_score, support = precision_recall_fscore_support(train_list_y[0:3000], y_pred_train, average=None)
        # print("precision", precision)
        # print("recall", recall)
        # print("f_score", f_score)
        # print("support", support)
        
        # ## score distribution (another view of the roc curve)
        # print("Check histogram?")
        # orig_pos_score = result[val_y == 0][:, 0]
        # orig_neg_score = result[val_y != 0][:, 0]
        # pos_score_histo = np.histogram(orig_pos_score, bins=5000)
        # neg_score_histo = np.histogram(orig_neg_score, bins=5000)
        # fig1 = plt.figure()
        # ax1 = plt.axes()
        # ax1.plot(neg_score_histo[1][:neg_score_histo[1].shape[0]-1], neg_score_histo[0], color='y')
        # ax1.set_xlim([0, 1])
        # ax1.set_ylim([0, 15])
        # fig = plt.gcf()
        # fig.set_size_inches(10.5, 15.5)
        # fig2 = plt.figure()
        # ax2 = plt.axes()
        # ax2.plot(pos_score_histo[1][:pos_score_histo[1].shape[0]-1], pos_score_histo[0], color='b')
        # ax2.set_xlim([0, 1])
        # ax2.set_ylim([0, 15])
        # fig = plt.gcf()
        # fig.set_size_inches(10.5, 15.5)