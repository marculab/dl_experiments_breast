from keras import regularizers
from keras import initializers
from keras import optimizers
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

PLOT_PATH = './reports'
LOG_PATH = './logs'

def iterate_samples(sample_num, val_num, test_num, val_options=3):
    ## generate all possible combinations: train, val, test
    ## for leave out validation
    samples = set(range(sample_num))
    test_indices = combinations(samples, test_num)
    indices = []
    ## choose only val_options combinations
    for tests in test_indices:
        rm_samples = samples.difference(set(tests))
        val_indices = np.array(list(combinations(rm_samples, val_num)))
        choice = np.random.choice(np.arange(val_indices.shape[0]), size=val_options, replace=False)
        for vals in val_indices[choice]:
            trains = np.array(list(rm_samples.difference(set(vals))))
            indices.append((trains, vals, np.array(tests)))

    return indices

def iterate_hyperparas(use_default=False):
    ## initializer, regularization, activation function, learning rate, batch size
    ## some parameters may change during training? (learning rate). ignore this need for now
    options = {}
    # defaults = {}
    # defaults['activation_regularizer'] = None
    # defaults['kernel_regularizer'] = 0.001
    # defaults['activation_regularizer'] = 0.1
    ## initializer
    options['initializer'] = []
    options['initializer'].append(initializers.glorot_normal(seed=None))
    options['initializer'].append(initializers.glorot_uniform(seed=None))
    options['initializer'].append(initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
    options['initializer'].append(initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None))
    options['initializer'].append(initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None))
    ## kernel_regularizer
    options['kernel_regularizer'] = []
    options['kernel_regularizer'].append(None)
    options['kernel_regularizer'].append(regularizers.l1(0.0001))
    options['kernel_regularizer'].append(regularizers.l1(0.001))
    options['kernel_regularizer'].append(regularizers.l1(0.01))
    options['kernel_regularizer'].append(regularizers.l1(0.1))
    options['kernel_regularizer'].append(regularizers.l2(0.0001))
    options['kernel_regularizer'].append(regularizers.l2(0.001))
    options['kernel_regularizer'].append(regularizers.l2(0.01))
    options['kernel_regularizer'].append(regularizers.l2(0.1))
    options['kernel_regularizer'].append(regularizers.l1_l2(0.0001))
    options['kernel_regularizer'].append(regularizers.l1_l2(0.001))
    options['kernel_regularizer'].append(regularizers.l1_l2(0.01))
    options['kernel_regularizer'].append(regularizers.l1_l2(0.1))
    ## activity_regularizer
    options['activity_regularizer'] = []
    options['activity_regularizer'].append(None)
    options['activity_regularizer'].append(regularizers.l1(0.0001))
    options['activity_regularizer'].append(regularizers.l1(0.001))
    options['activity_regularizer'].append(regularizers.l1(0.01))
    options['activity_regularizer'].append(regularizers.l1(0.1))
    options['activity_regularizer'].append(regularizers.l2(0.0001))
    options['activity_regularizer'].append(regularizers.l2(0.001))
    options['activity_regularizer'].append(regularizers.l2(0.01))
    options['activity_regularizer'].append(regularizers.l2(0.1))
    options['activity_regularizer'].append(regularizers.l1_l2(0.0001))
    options['activity_regularizer'].append(regularizers.l1_l2(0.001))
    options['activity_regularizer'].append(regularizers.l1_l2(0.01))
    options['activity_regularizer'].append(regularizers.l1_l2(0.1))
    ## bias_regularizer
    options['bias_regularizer'] = []
    options['bias_regularizer'].append(None)
    options['bias_regularizer'].append(regularizers.l1(0.0001))
    options['bias_regularizer'].append(regularizers.l1(0.001))
    options['bias_regularizer'].append(regularizers.l1(0.01))
    options['bias_regularizer'].append(regularizers.l1(0.1))
    options['bias_regularizer'].append(regularizers.l2(0.0001))
    options['bias_regularizer'].append(regularizers.l2(0.001))
    options['bias_regularizer'].append(regularizers.l2(0.01))
    options['bias_regularizer'].append(regularizers.l2(0.1))
    options['bias_regularizer'].append(regularizers.l1_l2(0.0001))
    options['bias_regularizer'].append(regularizers.l1_l2(0.001))
    options['bias_regularizer'].append(regularizers.l1_l2(0.01))
    options['bias_regularizer'].append(regularizers.l1_l2(0.1))
    ## activation
    options['activation'] = []
    options['activation'].append('relu')
    options['activation'].append('elu')
    options['activation'].append('selu')
    options['activation'].append('tanh')
    options['activation'].append('sigmoid')
    ## optimizer and learning rate
    options['optimizer'] = []
    options['optimizer'].append(optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False))
    options['optimizer'].append(optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))
    options['optimizer'].append(optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False))
    options['optimizer'].append(optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0))
    options['optimizer'].append(optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    options['optimizer'].append(optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    options['optimizer'].append(optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    options['optimizer'].append(optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004))
    options['optimizer'].append(optimizers.Nadam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004))
    options['optimizer'].append(optimizers.Nadam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004))
    ## batch size
    options['batch_size'] = 32
    options['batch_size'] = 128
    options['batch_size'] = 1024
    options['batch_size'] = 8192

    return options

def plot_accuracy(train_acc, val_acc):
    x = np.arange(len(train_acc)) + 1
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.plot(x, train_acc, x, val_acc)
    plt.legend(['train accuracy', 'val accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

def plot_loss(train_loss, val_loss):
    x = np.arange(len(train_loss)) + 1
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.plot(x, train_loss, x, val_loss)
    plt.legend(['train loss', 'val loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')

def plot_csv_log(logname):
    ## csv format: train loss, train accuracy, val loss, val accuracy
    logp = Path(LOG_PATH)
    plotp = Path(PLOT_PATH)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    with open(str(logp.joinpath(logname))) as infile:
        cur_p = plotp.joinpath(logname)
        if not cur_p.exists():
            cur_p.mkdir(parents=True)
        for i, line in enumerate(infile):
            if i == 0:
                continue
            stats_str = line.strip('\n').split(',')
            stats = [float(s.strip('\n')) for s in stats_str]
            train_loss.append(stats[0])
            train_acc.append(stats[1])
            val_loss.append(stats[2])
            val_acc.append(stats[3])
            ## plot loss
            plot_loss(train_loss, val_loss)
            plt.savefig(str(cur_p.joinpath('loss.png')))
            plt.clf()
            ## plot acc
            plot_loss(train_acc, val_acc)
            plt.savefig(str(cur_p.joinpath('accuracy.png')))
            plt.clf()

def generate_model_report(model):
    ## before build the model
    ## before compile the model, choose a optimizer
    ## fit model, apply other parameters
    pass

def main():
    idxs = iterate_samples(20, 2, 1)
    print(idxs)
    
    #options = iterate_hyperparas()
    #print(options)

if __name__ == '__main__':
    main()
    
