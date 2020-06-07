#!/usr/bin/env python
# coding: utf-8
# Alexander M. Yuan
# May 15, 2020
#
# June 2, 2020: merge transfer1 and transfer2
# June 1, 2020: merge training
# May 24, 2020: Change the output format to one-hot classification
# May 23, 2020: Add Bidirectional RNN option
# May 22, 2020: Merge 1 layer and 2 layer, with or without dropout
#
# May 20, 2020: converted to the new output with 1 value
#
# May 19, 2020: added XXXYYY2 to command line
# May 18, 2020: added TFILE2 to command line
# May 17, 2020: added TFILE to command line
# May 16, 2020: added batch_size and epoches to command line 
#
# To run the program:
#
# <draco> python3 Alex_gRNN_paper.py [SimpleRNN|LSTM|GRU] [acti] [n_a] [Tx] [predict_d] [Tstep] [Tcutoff] [Sstep] [batch_size] [epoches] [TFILE] [XXX] {YYY] [TFILE2]
#
# This implements the general RNN program in using Keras in the paper.
# The program is modified from Alex_RNN_Paper.py
#
# This is the jupyter program that is modified from Prof. Chakraborty's
# RNN for FoG detection for PD patience. Trying to reproduce the results
# in paper: 
#
# Vishwas G. Torvi, Aditya Bhattacharya, and Shayok Chakraborty, "Deep
# Domain Adaptation to Predict Freezing of Gait in Patients with Parkinson's
# Disease," IEEE ICMLA 2018.
#
# The baseline code is provided by Prof. Chakraborty. The code no longer
# works in the current version of python and tensorflow. Rewrote the code
# to generate X and Y arrays for training set and testing set, use Keras
# instead of tensorflow to build the RNN network.
# cell to import packages: numpy, matplotlib, tensorflow, keras,
# sklearn.metrics

# In[1]:


import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  

from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Bidirectional
from keras.layers import SimpleRNN, LSTM, GRU, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from sklearn import metrics
from numpy.random import seed
from tensorflow.random import set_seed

seed(1)
set_seed(2)

# In[2]:


# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "ankle_acc_x_",
    "ankle_acc_y_",
    "ankle_acc_z_",
    "upperleg_acc_x_",
    "upperleg_acc_y_",
    "upperleg_acc_z_",
    "trunk_acc_x_",
    "trunk_acc_y_",
    "trunk_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "NON-FREEZING",
    "FREEZING"
] 


# ## Preparing Dataset

# In[3]:


# file directory, 
#train data and test set are all in the directory
# 2/3 data for training, 1/3 data for testing

TFILE='S01/'
TFILE2=''
XXX=2
YYY=3
XXXYYY2 = 0
TRAIN = '../../dataset_fog_release/dataset/'
#TRAIN = '../../dataset_fog_release/dataset/test/'
#TRAIN = 'D:/FoG/dataset_fog_release/dataset/Train/'

# parameters
# prediction and RNN parameter, related to how to arrange the data
predict_d = 300  # how many samples in the forward FOG prediction
Tx = 50 # number of time steps in the RNN
Tstep = 1 # RNN examines data every Tstep entries

Tcutoff = 600000 # Examine at most 1500*0.015 seconds before FoG
Sstep = 1 # sample step is 6
n_values = 9
n_a=1 # RNN internal size
af = 'tanh'
batch_size = 128
epoches = 5
RNNdropout = 0.0
RNNlayer = 2
RNNfunc = 'normal'
RNNcellname = 'SimpleRNN'
datasettype = 'under'
classratio = 1.2

RNN_cell1 = SimpleRNN(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
RNN_cell2 = SimpleRNN(n_a, activation=af)
RNN_cellX = SimpleRNN(n_a, activation=af, input_shape=(Tx, n_values))

RNN_cellXX = SimpleRNN(n_a, activation=af, return_sequences=True)

def parse_commandline():
    global RNN_cell1, RNN_cell2, RNN_cellX, af, Tx, predict_d, n_a, Tstep, Tcutoff, Sstep
    global batch_size
    global epoches
    global TRAIN
    global TFILE
    global XXX
    global YYY
    global TFILE2, RNN_cellXX
    global XXXYYY2, RNNdropout, RNNlayer, RNNfunc, RNNcellname
    global datasettype, classratio

    if len(sys.argv) >2:
        af = sys.argv[2]

    if len(sys.argv) > 3:
        n_a = int (sys.argv[3])

    if len(sys.argv) > 4:
        Tx = int (sys.argv[4])

    if len(sys.argv) > 5:
        predict_d = int (sys.argv[5])

    if len(sys.argv) > 6:
        Tstep = int (sys.argv[6])

    if len(sys.argv) > 7:
        Tcutoff = int (sys.argv[7])

    if len(sys.argv) > 8:
        Sstep = int (sys.argv[8])

    if len(sys.argv) > 9:
        batch_size = int (sys.argv[9])

    if len(sys.argv) > 10:
        epoches = int (sys.argv[10])

    if len(sys.argv) > 11:
        TFILE = sys.argv[11]

    if len(sys.argv) > 12:
        XXX = int(sys.argv[12])

    if len(sys.argv) > 13:
        YYY = int(sys.argv[13])

    if len(sys.argv) > 14:
        TFILE2 = sys.argv[14]

    if len(sys.argv) > 15:
        XXXYYY2 = int(sys.argv[15])

    if len(sys.argv) > 16:
        RNNdropout = float(sys.argv[16])

    if len(sys.argv) > 17:
        RNNlayer = int(sys.argv[17])
        if RNNlayer != 1: 
            RNNlayer = 2

    if len(sys.argv) > 18:
        RNNfunc = sys.argv[18]

    if len(sys.argv) > 19:
        datasettype = sys.argv[19]

    if len(sys.argv) > 20:
        classratio = float(sys.argv[20])

    
    if len(sys.argv) > 1:
        RNNcellname = sys.argv[1]
        if sys.argv[1] == 'SimpleRNN' :
            RNN_cell1 = SimpleRNN(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cellX = SimpleRNN(n_a, activation=af, input_shape=(Tx, n_values))
            RNN_cellXX = SimpleRNN(n_a, activation=af, return_sequences=True)
            RNN_cell2 = SimpleRNN(n_a, activation=af)
        elif sys.argv[1] == 'LSTM' :
            RNN_cell1 = LSTM(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = LSTM(n_a, activation=af)
            RNN_cellXX = LSTM(n_a, activation=af, return_sequences=True)
            RNN_cellX = LSTM(n_a, activation=af, input_shape=(Tx, n_values))
        elif sys.argv[1] == 'GRU' :
            RNN_cell1 = GRU(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = GRU(n_a, activation=af)
            RNN_cellXX = GRU(n_a, activation=af, return_sequences=True)
            RNN_cellX = GRU(n_a, activation=af, input_shape=(Tx, n_values))
        elif sys.argv[1] == 'BSimpleRNN' :
            RNN_cell1 = SimpleRNN(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = SimpleRNN(n_a, activation=af)
            RNN_cellX = SimpleRNN(n_a, activation=af, input_shape=(Tx, n_values))
            RNN_cellXX = SimpleRNN(n_a, activation=af, return_sequences=True)
            RNN_cellXX = Bidirectional(RNN_cellXX)
            RNN_cellX = Bidirectional(RNN_cellX)
            RNN_cell1 = Bidirectional(RNN_cell1)
            RNN_cell2 = Bidirectional(RNN_cell2) 

        elif sys.argv[1] == 'BLSTM' :
            RNN_cell1 = LSTM(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = LSTM(n_a, activation=af)
            RNN_cellX = LSTM(n_a, activation=af, input_shape=(Tx, n_values))
            RNN_cellX = Bidirectional(RNN_cellX)
            RNN_cellXX = LSTM(n_a, activation=af, return_sequences=True)
            RNN_cellXX = Bidirectional(RNN_cellXX)
            RNN_cell1 = Bidirectional(RNN_cell1)
            RNN_cell2 = Bidirectional(RNN_cell2) 
        elif sys.argv[1] == 'BGRU' :
            RNN_cell1 = GRU(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = GRU(n_a, activation=af)
            RNN_cellX = GRU(n_a, activation=af, input_shape=(Tx, n_values))
            RNN_cellX = Bidirectional(RNN_cellX)
            RNN_cellXX = GRU(n_a, activation=af, return_sequences=True)
            RNN_cellXX = Bidirectional(RNN_cellXX)
            RNN_cell1 = Bidirectional(RNN_cell1)
            RNN_cell2 = Bidirectional(RNN_cell2) 
        else :
            print('Usage: python3 Alex_gRNN_paper.py [(B)SimpleRNN|(B)LSTM|(B)GRU] [acti] [n_a] [Tx] [predict_d] [Tstep] [Tcutoff] [Sstep] [batch_size] [epoches] [TFILE] [XXX] [YYY] [TFILE2] [XXXYYY2] [RNNDrop] [RNNLayer] [RNNfunc] [datasettype] [classratio]')
            exit(0)

    print('Training with the following parameters: ')
    print('  RNNcellname =', RNNcellname)
    print('  RNN_Cell1 = ', RNN_cell1)
    print('  RNN_Cell2 = ', RNN_cell2)
    print('  Activation func = ', af)
    print('  Tx = ', Tx)
    print('  n_a = ', n_a)
    print('  predict_d = ', predict_d)
    print('  Tstep = ', Tstep)
    print('  Tcutoff = ', Tcutoff)
    print('  Sstep = ', Sstep)
    print('  batch_size = ', batch_size)
    print('  epoches = ', epoches)
    print('  TRAIN directory =', TRAIN + TFILE)
    print('  percentage XXX =', XXX)
    print('  percentage YYY =', YYY)

    if TFILE2 =='':
        print('  No separate testing directory.')
    else : 
        print('  Testing directory = ', TRAIN+TFILE2)
        if XXXYYY2 != 0 :
            print('  Testing directory uses percentage.')
        else :
            print('  Testing directory use all teseting.')

    print('  RNNdropout = ', RNNdropout)
    print('  RNNlayer = ', RNNlayer)
    print('  RNNfunc = ', RNNfunc)
            
parse_commandline()
    
# ## New code to load data Data Loading

# In[4]:


# the program is modified from Prof. Chakarborty's program
#
# create the X and Y values from the files 
#
#.  1. remove entries not in the experiments 
#.  2. Experimental data are partitioned into episodes
#.  3. Generate batch of Tx entries for each episode (X) 
#.     with marking (Y), and concatenate all data for 
#.     training and testing.

e_count = 0 # count the number of episodes

def shuffle_list(a, b):
    assert len(a)==len(b)
    s_a = []
    s_b = []
    for i in np.random.permutation(len(a)):
        s_a += [a[i]]
        s_b += [b[i]]
    return s_a, s_b

# generate X, Y for one episode of Fog or end of file
def gen_batch_one_episode(data, beg, end, shuffle = 1):
    X = []
    Y = []
    XX = []
    YY = []
    i = end - 1;
    while data[i][9] == '2':
        i -= 1
    
    # cut off if necessary
    if i-Tcutoff > beg:
        beg = i-Tcutoff
    
    if beg > (Tx-1)*Tstep : 
        rbeg = beg - (Tx-2)* Tstep
    else:
        rbeg = beg

    for i in [n for n in range(rbeg, end-Tx*Tstep, Sstep)]:
        t = []

        for j in range(Tx):
            if i+j*Tstep >= beg : 
                t += [data[i+j*Tstep][:9]]
            else :
                t += [[0, 0, 0, 0, 0, 0, 0, 0, 0]] 

        X += [t]
        if i+(Tx-1)*Tstep+predict_d > end-1:
            if data[end-1][9] == '2' or data[i+(Tx-1)*Tstep][9] == '2':
                Y += [[0,1]]
            else :
                Y += [[1, 0]]
        else :
            if data[i+(Tx-1)*Tstep+predict_d][9] == '2' or data[i+(Tx-1)*Tstep][9] == '2':
                Y += [[0, 1]]
            else :
                Y += [[1, 0]]
       
    if shuffle != 1:
        return X, Y
    
    X1, Y1 = shuffle_list(X, Y)
    
    return X1, Y1

# generate X, Y for all episode
def gen_batch(data, shuffle = 1):
    global e_count
    X = []
    Y = []
    X1 = []
    Y1 = []
    beg = 0
    end = beg
    end1 = 0
    print('len=', len(data))
    while beg < len(data):
        end = beg
        while end < len(data) and data[end][9] != '2':
            # print(end, data[end], '**', data[end][9])
            end = end + 1
        end1 = end
        while end < len(data) and data[end][9] == '2':
            end = end + 1
        if data[end-1][9] == '2': 
            e_count = e_count+1            
            X1, Y1 = gen_batch_one_episode(data, beg, end, shuffle = shuffle)
            X += X1
            Y += Y1
#        print(beg, ' ', end1, ' ', end, ' ', end1-beg, ' ', end-end1)
        beg = end
    return X, Y
        
def load_data(file_paths, shuffle = 1):
    global e_count
    train = []
    test = []
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    batch_size = 0
    for root, dirs, filenames in os.walk(file_paths):
        for fffDir in filenames:
            X_signals = []
            file = open(file_paths + fffDir, 'r')
            for row in file:
                batch_signals = []
#                print(row.strip().split(' ')[1:])
                if int(row.strip().split(' ')[1:][9]) != 0:
                    batch_signals += [row.strip().split(' ')[1:]]
                    X_signals += batch_signals
            file.close()
            
            train += X_signals[:XXX*int(len(X_signals)/YYY)]
            test += X_signals[XXX*int(len(X_signals)/YYY):]
            #train += X_signals[len(X_signals)-120:len(X_signals)-60]
            #test += X_signals[len(X_signals)-60:]
    
    train = np.array(train)
    train_X, train_Y = gen_batch(train, shuffle = shuffle)
    print(e_count, ' episodes in training data')
    e_count = 0
    test_X, test_Y = gen_batch(test, shuffle = shuffle)
    print(e_count, ' episodes in testing data')
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_data(TRAIN+TFILE, shuffle = 0)
if TFILE2 != '' and TFILE2 != TFILE :
    train_X += test_X
    train_Y += test_Y
    train_X1, train_Y1, test_X, test_Y = load_data(TRAIN+TFILE2, shuffle = 0)
    if XXXYYY2 == 0: 
        test_X = train_X1 + test_X
        test_Y = train_Y1 + test_Y
    else :
        train_X = train_X + train_X1
        train_Y = train_Y + train_Y

    if RNNfunc == 'transfer1' or RNNfunc == 'transfer2' or RNNfunc == 'transfer3':
        train_X = train_X1
        train_Y = train_Y1

def remake_training_set(X, Y, datasettype, ratio) :
    XX0 = []
    XX1 = []
    for i in range(len(X)) : 
        if Y[i][0] == 0 :
            XX1 += [X[i]]
        else : 
            XX0 += [X[i]]

    if datasettype == 'under' :
        XX = []
        YY = []
        random.shuffle(XX0)
        if len(XX0) > ratio * len(XX1) :
            a = ratio*len(XX1)
        else :
            a = len(XX0)

        for i in range(int(a)) :
            XX += [XX0[i]]
            YY += [[1, 0]]
        for i in range(len(XX1)) :
            XX += [XX1[i]]
            YY += [[0, 1]]
        XX, YY = shuffle_list(XX, YY)
        return XX, YY
    elif datasettype == 'over' :
        XX = []
        YY = []
        random.shuffle(XX1)
        for i in range(len(XX0)) :
            XX += [XX0[i]]
            YY += [[1, 0]]
        for i in range(int(len(XX0)/ratio)) :
            XX += [XX1[i % len(XX1)]]
            YY += [[0, 1]]
        XX, YY = shuffle_list(XX, YY)
        return XX, YY
    else :
        return X, Y

def remake_test_set(X, Y) :
    return X, Y

train_X, train_Y = remake_training_set(train_X, train_Y, datasettype, classratio)
test_X, test_Y = remake_test_set(test_X, test_Y)

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
#train_Y = np.reshape(train_Y, (len(train_Y), 1))
test_Y = np.array(test_Y)
#test_Y = np.reshape(test_Y, (len(test_Y), 1))


print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

c = 0;
for j in range(train_X.shape[0]):
    if train_Y[j][0] == 0:
        c += 1

print('train_Y count = ', c)

c = 0;
for j in range(test_X.shape[0]):
    if test_Y[j][0] == 0 :
        c += 1

print('test_Y count = ', c)

# In[5]:


#sample = open('output.txt', 'w')
#np.set_printoptions(threshold=sys.maxsize)
#print(train_Y, file=sample)
#sample.close()


# # Build the RNN
#    Code is copied and modified from the coursera course

# In[6]:


n_values = 9
reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(2, activation='softmax')


# In[7]:


def mymodel(Tx, n_a, n_values): 
    """
    Implement the model
    Arguments: 
    Tx -- length of the time steps
    n_a -- the number of activations in the model
    n_values -- number of values as the input per time step
    
    Returns: 
    model -- a keras instance model with n_a activations
    """
    
    #define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    #define the initial hidden state a0
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Step 1: Create empty list to append the outputs while you iterate
    #outputs = []
    
    # Step 2: Loop
    
    #for t in range(Tx):
        
    #    x = Lambda(lambda z: z[:, t, :])(X)
    #    x = reshapor(x)
    #    a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
    #    if t==Tx-1:
    #         out=densor(a)
    #         outputs.append(out)
    #print(outputs)
    # Step 3: Create model instance
    #model = Model(inputs=[X, a0, c0], outputs=outputs)

    model = Sequential()
    if RNNlayer == 2: 
        model.add(RNN_cell1)
        if RNNdropout > 0:     
            model.add(Dropout(RNNdropout))
        model.add(RNN_cell2)
        if RNNdropout > 0: 
            model.add(Dropout(RNNdropout))
        model.add(densor)
    elif RNNlayer == 1:
        model.add(RNN_cellX)
        if RNNdropout > 0:     
            model.add(Dropout(RNNdropout))
        model.add(densor)              
    return model



def print_metrics(RealY, PredY) :
    total = RealY.shape[0]
    c0=0
    c1=0

    tp0 = 0
    fn0 = 0
    tp1 = 0
    fn1 = 0

    for i in range(total):
        if RealY[i][0] == 1:
            c0 += 1
            if PredY[i][0] == 1:
                tp0 += 1
            else : 
                fn0 += 1
        else :
            c1 += 1
            if PredY[i][0] == 0:
                tp1 += 1
            else :
                fn1 += 1

    accuracy = (tp0+tp1)/total
    sensitivity = tp1/c1
    specificity = tp0/c0
    gmean = np.sqrt(sensitivity*specificity)
    if tp1+fn0 != 0:
        precision = tp1 / (tp1+fn0)
    else :
        precision = 0;

    recall = sensitivity
    if precision+recall != 0 :
        fmeasure = (2*precision*recall)/(precision+recall)
    else :
        fmeasure = 0

    print ('  total cases: ', total, ', negative cases: ', c0, ', positive cases:', c1)
    print ('  negative cases, true positive: ', tp0, ', false negative: ', fn0)
    print ('  postive  cases, true positive: ', tp1, ', false negative: ', fn1)
    print ('  accuracy: ', accuracy)
    print ('  sensitivity: ', sensitivity)
    print ('  specificity: ', specificity)
    print ('  G-mean: ', gmean)
    print ('  precision : ', precision)
    print ('  Recall: ', recall)
    print ('  f-measure: ', fmeasure)

def print_metrics_file(RealY, PredY, f) :
    total = RealY.shape[0]
    c0=0
    c1=0

    tp0 = 0
    fn0 = 0
    tp1 = 0
    fn1 = 0

    for i in range(total):
        if RealY[i][0] == 1:
            c0 += 1
            if PredY[i][0] == 1:
                tp0 += 1
            else : 
                fn0 += 1
        else :
            c1 += 1
            if PredY[i][0] == 0:
                tp1 += 1
            else :
                fn1 += 1

    accuracy = (tp0+tp1)/total
    sensitivity = tp1/c1
    specificity = tp0/c0
    gmean = np.sqrt(sensitivity*specificity)
    if tp1+fn0 != 0:
        precision = tp1 / (tp1+fn0)
    else :
        precision = 0;

    recall = sensitivity
    if precision+recall != 0 :
        fmeasure = (2*precision*recall)/(precision+recall)
    else :
        fmeasure = 0

    print (file=f, '  total cases: ', total, ', negative cases: ', c0, ', positive cases:', c1)
    print (file=f, '  negative cases, true positive: ', tp0, ', false negative: ', fn0)
    print (file=f, '  postive  cases, true positive: ', tp1, ', false negative: ', fn1)
    print (file=f, '  accuracy: ', accuracy)
    print (file=f, '  sensitivity: ', sensitivity)
    print (file=f, '  specificity: ', specificity)
    print (file=f, '  G-mean: ', gmean)
    print (file=f, '  precision : ', precision)
    print (file=f, '  Recall: ', recall)
    print (file=f, '  f-measure: ', fmeasure)



# In[8]:

parse_commandline()
model = mymodel(Tx=Tx, n_a=n_a, n_values = 9)
#model.summary()

# In[9]:

opt = Adam(lr=0.0009, beta_1 = 0.9, beta_2=0.999, decay = 0.01)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# # initialize hidden state and cell state

# In[10]:

XXX=train_X
YYY = train_Y
print(YYY.shape)

# In[11]:

if RNNfunc != 'transfer1' and RNNfunc != 'transfer2' and RNNfunc != 'transfer3' and RNNfunc != 'load' :
    model.fit(XXX, YYY, batch_size=batch_size, epochs=epoches)
    model.summary()

if RNNfunc == 'train':
    model.save(RNNcellname+'_'+TFILE[:-1]+'_'+af+'_'+str(predict_d)
               +'_'+str(Tx)+'_'+str(n_a)+'_'+str(RNNdropout)+'_'+str(RNNlayer)+'.h5')
    exit(0)

if RNNfunc == 'load' or RNNfunc == 'transfer1' :  
    model = load_model('model/'+RNNcellname+'_'+TFILE[:-1]+'_'+af+'_'+str(predict_d)
               +'_'+str(Tx)+'_'+str(n_a)+'_'+str(RNNdropout)+'_'+str(RNNlayer)+'.h5')
    if RNNfunc == 'transfer1' : 
        Oldout = model.predict(test_X, verbose=0)
        Oldout1 = np.around(Oldout)
        print('Stat for test set (before transfer1): ')
        print_metrics(test_Y, Oldout1)
        model.fit(train_X, train_Y, batch_size=batch_size, epochs=epoches)
        model.summary()
elif RNNfunc == 'transfer2' or RNNfunc == 'transfer3': # this only does the two layer case in the paper
    model1 = Sequential()
    model1.add(RNN_cell1)
    model1.add(RNN_cellXX)
    model1.add(densor)
    model1.load_weights('model/'+RNNcellname+'_'+TFILE[:-1]+'_'+af+'_'+str(predict_d)
               +'_'+str(Tx)+'_'+str(n_a)+'_'+str(RNNdropout)+'_'+str(RNNlayer)+'.h5')

    model = Sequential()
    for l in model1.layers[:-1] :
        model.add(l)
    model.add(RNN_cell2)
    model.add(densor)

    if RNNfunc == 'transfer2' :
        model.layers[0].trainable = False
        model.layers[1].trainable = False
    for l in model.layers:
        print(l.name, l.trainable)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    Oldout = model.predict(test_X, verbose=0)
    Oldout1 = np.around(Oldout)
    print('Stat for test set (before transfer2): ')
    print_metrics(test_Y, Oldout1)
    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epoches)
    model.summary()

# In[12]:

#model.save(sys.argv[1]+'_'+TFILE+'.h5')

#model1 = load_model('rnn_model.h5')
Out = model.predict(XXX, verbose=0)

# ## try the model on the train and test data

# In[13]:

Out1 = np.around(Out)
print(Out1.shape[0])
print(Out1.shape)

print('Stat for training set:')
print_metrics(YYY, Out1)

# In[14]:

NXXX=test_X
NYYY=test_Y

NOut = model.predict(NXXX, verbose=0)
NOut1 = np.around(NOut)

print('Stat for test set:')
print_metrics(NYYY, NOut1)

def print_to_file(filename, RealY, PredY, RPredY) :
    sample = open(filename, 'w')
    c = 0
    i = 0;
    fog = []
    fogc = 0
    pre = []
    s = []
    print_metrics_file(RealY, RPredY, sample)

    while i < RealY.shape[0] :
        while i < RealY.shape[0] and RealY[i][0] == 1:
            i += 1
        i+=predict_d
        a = i;
        while i < RealY.shape[0] and RealY[i][0] == 0:
            i += 1
        fog += [[a, i]]
        c += 1

    fogc = c
    i = 0

    while i < RealY.shape[0]/30 :
        ss = 0
        for j in range(30):
            if i*30+j < RealY.shape[0] :
                ss += RPredY[i*30+j][1]
            else :
                ss += RPredY[-1][1]
        if ss >= 15: 
            s += [1]
        else :
            s += [0]
        i += 1

    i = 0
    c = 0

    fogi = 0
    al = []
    fa = 0
    fal = []

    while i < len(s) :
        while i < len (s) and s[i] == 0 :
            i += 1
        beg = i*30
        flag = 1
        while flag == 1:
            while i < len (s) and s[i] == 1 :
                i += 1
            if i+1 < len (s) and s[i+1] == 1 :
                i += 1
            else :
                flag = 0
        end = i*30
        al += [[beg, end]]
        
        if fogi < len(fog) and fog[fogi][0] > end :
            fa += 1
            fal += [[beg, end]]
        else :    
            while fogi < len(fog) and fog[fogi][1] < end :
                if fog[fogi][1] < beg :
                    pre += [10000]
                    fogi += 1
                else :
                    pre += [fog[fogi][0] - beg]
                    fogi += 1
            if fogi < len(fog) and fog[fogi][0] < end : 
                pre += [fog[fogi][0] - beg]
                fogi += 1
                
    for i in range (len(fog) - len(pre)) :
        pre += [1000]
        

    print('A totle of ', fogc, ' FoG episodes.', file = sample)

    for i in range(len(fog)) :
        print('  episode ', i, ': (', fog[i][0], ', ', fog[i][1], '), predicted ', pre[i],
              ' time units (0.015s) ahead', file=sample)

    print('A total of ', len(al), ' alarms', file=sample)
    for i in range(len(al)) :
        print('  alarm ', i, ': (', al[i][0], ', ', al[i][1], ')', file=sample)

    print('A total of ', fa, ' false alarms', file=sample)
    for i in range(len(fal)) :
        print('  false alarm ', i, ': (', fal[i][0], ', ', fal[i][1], ')', file=sample)

    print('Alarm :', file=sample)
    for i in range(len(s)) :
        if s[i] == 1 :
            print(i, ' : yes', file=sample)
        else :
            print(i, ': no', file = sample)



    for i in range (RealY.shape[0]):
        if (RealY[i][0] == 0 and PredY[i][0] < 0.5) or (RealY[i][1] == 0 and PredY[i][0] >= 0.5)  :
            print(i, ': (', RealY[i][0], ', ', RealY[i][1], 
                  '), yes, (', PredY[i][0], ', ', PredY[i][1], ')', file=sample)
        else : 
            print(i, ': (', RealY[i][0], ', ', RealY[i][1], 
                  '),  no, (', PredY[i][0], ', ', PredY[i][1], ')', file=sample)

    sample.close()
  
print_to_file('output.txt', NYYY, NOut, NOut1)

AAAA = []
BBBB = []
CCCC = []
DDDD = []

#for i in range(NOut1.shape[0]):
#    if i<predict_d:
#        CCCC += [0]
#    else : 
#        CCCC += [NYYY[i]]
#
#    if NYYY[i] == 0:
#        AAAA += [1]
#    else :
#        AAAA += [0]
#
#    if NOut1[i] == 0:
#        BBBB += [1]
#    else :
#        BBBB += [0]
#
#    if i < predict_d:
#        DDDD += [0]
#    else :
#        jj = 0
#        for j in range(predict_d)[:-1] :
#            jj = (1-0.1*jj) + 0.1 * NOut1[i-j]
#        DDDD += [(jj*2 / predict_d)]

#plt.plot(range(NOut1.shape[0]), CCCC)
#plt.savefig('raw.png')
#
#plt.plot(range(NOut1.shape[0]), DDDD)
#plt.savefig('shift.png')
#
#plt.plot(range(NOut1.shape[0]), BBBB)
#plt.savefig('pred.png')

# In[15]:

#model.save('rnn_model_100_2_128_20.h5')

