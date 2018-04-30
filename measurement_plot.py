'''
Generate measures and plots that might be useful to tell the performance of a classifier

Log: (logs should be implemented by Callback class)
    * parameters of the classifier
    * y_train, y_val, train_score, val_score
    * accuracy (can be computed), loss

-Crossvalidation
    -Run
        -Epoch
            * precision, recall, F-score
            * loss, accuracy change (train/validation)
            * ROC curve
            * Score disribution (validation/ train?)
'''


import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from pathlib import Path
from experiment_utils import LogDir

class MeasurePlot():
    ## load the training, validation and prediction labels
    def __init__(self, y_train=None, y_val=None, log_root='../logs'):
        self.y_train = y_train
        self.y_val = y_val
        self.train_predict_score = None
        self.val_predict_score = None
        self.log_root = log_root
        self.log_dir = ''
        self.run_dir = ''
        self.epoch_dir = ''
        logdir = LogDir()
        self.Y_TRAIN_FILE = logdir.Y_TRAIN_FILE
        self.Y_VAL_FILE = logdir.Y_VAL_FILE
        self.TRAIN_PREDICT_SCORE_FILE = logdir.TRAIN_PREDICT_SCORE_FILE
        self.VAL_PREDICT_SCORE = logdir.VAL_PREDICT_SCORE_FILE


    def iterate_log_dir(self, log_dir, run_dir=None):
        ## main function, processing all the folders in a log_dir
        self.log_dir = Path(self.log_root).joinpath(log_dir)
        if run_dir == None: ## run_dir == None: go through all runs
            run_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]
            for this_run_dir in run_dirs:
                self.handle_one_run(this_run_dir)    
        else:
            this_run_dir = self.log_dir.joinpath(run_dir)
            self.handle_one_run(this_run_dir)

    def handle_one_run(self, this_run_dir):
        self.run_dir = this_run_dir
        self.y_train = np.load(str(this_run_dir.joinpath(self.Y_TRAIN_FILE)))
        self.y_val = np.load(str(this_run_dir.joinpath(self.Y_VAL_FILE)))
        train_loss_arr = []
        train_acc_arr = []
        val_loss_arr = []
        val_acc_arr = []
        ## iterate epochs
        epoch_dirs = [d for d in this_run_dir.iterdir() if d.is_dir()]
        for this_epoch_dir in epoch_dirs:
            self.epoch_dir = this_epoch_dir
            self.train_predict_score = np.load(str(this_epoch_dir.joinpath(self.TRAIN_PREDICT_SCORE_FILE)))
            self.val_predict_score = np.load(str(this_epoch_dir.joinpath(self.VAL_PREDICT_SCORE)))
            self.plot_val_roc()
            self.plot_val_histogram()
            precision, recall, f_score = self.val_precision_recall_fscore()

    def plot_val_roc(self, pos_label=0, save=False, savedir=None):
        if savedir == None:
            savedir = self.run_dir
        count = 10
        pos_score = self.val_predict_score[:][:, pos_label]
        fpr, tpr, thresholds = roc_curve(self.y_val, pos_score, pos_label=pos_label)
        if thresholds.shape[0] < count:
            print("thresholds", thresholds)
        print("thresholds", thresholds[np.arange(0, thresholds.shape[0], int(thresholds.shape[0]/count))])
        fig = plt.figure()
        ax1 = plt.axes()
        ax1.plot(fpr, tpr)
        ax1.set_ylabel("True positive rate")
        ax1.set_xlabel("False positive rate")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        fig.set_size_inches(5, 5)

    def plot_val_histogram(self, pos_label=0, bins=5000, save=False, savedir=None):
        if savedir == None:
            savedir = self.run_dir
        orig_pos_score = self.val_predict_score[self.y_val == pos_label][:, pos_label]
        orig_neg_score = self.val_predict_score[self.y_val != pos_label][:, pos_label]
        pos_score_histo = np.histogram(orig_pos_score, bins=bins)
        neg_score_histo = np.histogram(orig_neg_score, bins=bins)
        f, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=True)
        ax1.plot(pos_score_histo[1][:neg_score_histo[1].shape[0]-1], pos_score_histo[0], color='tab:orange')
        ax2.plot(neg_score_histo[1][:neg_score_histo[1].shape[0]-1], neg_score_histo[0], color='tab:blue')
        ax1.set_xlim([0, 1.01])
        ax2.set_xlim([0, 1.01])
        ax1.set_ylim( [0, np.max([np.max(pos_score_histo[0]), np.max(neg_score_histo[0])]) + 0.5] )
        ax1.set_ylabel("count")
        ax2.set_ylabel("count")
        ax2.set_xlabel("score (probability to be cancer)")

    def val_precision_recall_fscore(self, save=False, savedir=None):
        if savedir == None:
            savedir = self.run_dir
        result = self.val_predict_score
        val_y_pred = np.apply_along_axis(np.argmax, axis=1, arr=result)
        precision, recall, f_score, support = precision_recall_fscore_support(self.y_val, val_y_pred, average=None)
        ## write to file, append at last line
        print("val_precision_recall_fscore", precision, recall, f_score)
        return precision, recall, f_score