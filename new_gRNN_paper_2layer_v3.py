#!/usr/bin/env python
# coding: utf-8
# Alexander M. Yuan
# May 15, 2020
#
# May 21, 2020: change to bidirectional RNN
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
#
# This is the jupyter program that is modified from Prof. Chakraborty's
# RNN for FoG detection for PD patience. Trying to reproduce the results
# in paper: 

# Vishwas G. Torvi, Aditya Bhattacharya, and Shayok Chakraborty, "Deep
# Domain Adaptation to Predict Freezing of Gait in Patients with Parkinson's
# Disease," IEEE ICMLA 2018.

# The baseline code is provided by Prof. Chakraborty. The code no longer
# works in the current version of python and tensorflow. Rewrote the code
# to generate X and Y arrays for training set and testing set, use Keras
# instead of tensorflow to build the RNN network.
# cell to import packages: numpy, matplotlib, tensorflow, keras,
# sklearn.metrics

# In[1]:


import os
import sys
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

RNN_cell1 = SimpleRNN(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
RNN_cell2 = SimpleRNN(n_a, activation=af)

def parse_commandline():
    global RNN_cell1, RNN_cell2, af, Tx, predict_d, n_a, Tstep, Tcutoff, Sstep
    global batch_size
    global epoches
    global TRAIN
    global TFILE
    global XXX
    global YYY
    global TFILE2
    global XXXYYY2

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

    if len(sys.argv) > 1:
        if sys.argv[1] == 'SimpleRNN' :
            RNN_cell1 = SimpleRNN(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = SimpleRNN(n_a, activation=af)
        elif sys.argv[1] == 'LSTM' :
            RNN_cell1 = LSTM(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = LSTM(n_a, activation=af)
        elif sys.argv[1] == 'GRU' :
            RNN_cell1 = GRU(n_a, activation=af, return_sequences=True, input_shape=(Tx, n_values))
            RNN_cell2 = GRU(n_a, activation=af)
        else :
            print('Usage: python3 Alex_gRNN_paper.py [SimpleRNN|LSTM|GRU] [acti] [n_a] [Tx] [predict_d] [Tstep] [Tcutoff] [Sstep] [batch_size] [epoches] [TFILE] [XXX] [YYY] [TFILE2] [XXXYYY2]')
            exit(0)

    print('Training with the following parameters: ')
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
        rbeg = beg - (Tx-2)*Tstep
    else :
        rbeg = beg
    
    for i in [n for n in range(rbeg, end-Tx*Tstep, Sstep)]:
        t = []

        for j in range(Tx):
            t += [data[i+j*Tstep][:9]]

        X += [t]
        if i+(Tx-1)*Tstep+predict_d > end-1:
            if data[end-1][9] == '2' or data[i+(Tx-1)*Tstep][9] == '2':
                Y += [1]
            else :
                Y += [0]
        else :
            if data[i+(Tx-1)*Tstep+predict_d][9] == '2' or data[i+(Tx-1)*Tstep][9] == '2':
                Y += [1]
            else :
                Y += [0]

    if shuffle != 1:
        return X, Y
    
    X1, Y1 = shuffle_list(X, Y)
    
    #for i in range(len(X)):
    #    if Y[i] == [0, 1]:
    #        XX += [X[i]]
    #        YY += [Y[i]]
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
if TFILE2 != '':
    train_X += test_X
    train_Y += test_Y
    train_X1, train_Y1, test_X, test_Y = load_data(TRAIN+TFILE2, shuffle = 0)
    if XXXYYY2 == 0: 
        test_X = train_X1 + test_X
        test_Y = train_Y1 + test_Y
    else :
        train_X = train_X + train_X1
        train_Y = train_Y + train_X2

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
    if train_Y[j] == 1:
        c += 1

print('train_Y count = ', c)

c = 0;
for j in range(test_X.shape[0]):
    if test_Y[j] == 1:
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
densor = Dense(1, activation='sigmoid')


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
    model.add(Bidirectional(RNN_cell1))
#    model.add(Dropout(0.7))
    model.add(Bidirectional(RNN_cell2))
#    model.add(Dropout(0.7))
    model.add(densor)              
    return model


# In[8]:


parse_commandline()
model = mymodel(Tx=Tx, n_a=n_a, n_values = 9)


# In[9]:


opt = Adam(lr=0.0009, beta_1 = 0.9, beta_2=0.999, decay = 0.01)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# # initialize hidden state and cell state

# In[10]:


XXX=train_X
YYY = train_Y

a0 = np.zeros((len(XXX), n_a))
c0= np.zeros((len(XXX), n_a))
print(YYY.shape)

# In[11]:

#myYYY = to_categorical(YYY)

model.fit(XXX, YYY, batch_size=batch_size, epochs=epoches)

model.summary()

# In[12]:

#model.save('rnn_model.h5')

#model1 = load_model('rnn_model.h5')
Out = model.predict(XXX, verbose=0)

# ## try the model on the train and test data

# In[13]:

Out1 = np.around(Out)
print(Out1.shape[0])
print(Out1.shape)

c0=0
c1=0
c2=0
for i in range(Out1.shape[0]):
    if YYY[i] == 1:
        c0 += 1
    if Out1[i] != YYY[i]:
        if Out1[i] == 0:
            c1 += 1
        else : 
            c2 += 1
            
print('c0 = ', c0)
print('c1 = ', c1)
print('c2 = ', c2)

# In[14]:


NXXX=test_X
NYYY=test_Y

a1 = np.zeros((len(NXXX), n_a))
c1= np.zeros((len(NYYY), n_a))
print(NYYY.shape)
NOut = model.predict(NXXX, verbose=0)

NOut1 = np.around(NOut)
print(NOut1.shape[0])
print(NOut1.shape)

Nc0=0
Nc1=0
Nc2=0
for i in range(NOut1.shape[0]):
    if NYYY[i] == 1:
        Nc0 += 1
    if NOut1[i] != NYYY[i]:
        if NOut1[i] == 0:
            Nc1 += 1
        else : 
            Nc2 += 1
            
print('Nc0 = ', Nc0)
print('Nc1 = ', Nc1)
print('Nc2 = ', Nc2)

print("accuracy_score = {}%".format(100*metrics.accuracy_score(NYYY, NOut1, normalize=True))) 
print("precision_score = {}%".format(100*metrics.precision_score(NYYY, NOut1, average="weighted"))) 
print("Recall_score = {}%".format(100*metrics.recall_score(NYYY, NOut1, average="weighted")))
print("f1_score = {}%".format(100*metrics.f1_score(NYYY, NOut1, average="weighted")))

AAAA = []
BBBB = []
CCCC = []
DDDD = []

for i in range(NOut1.shape[0]):
    if i<predict_d:
        CCCC += [0]
    else : 
        CCCC += [NYYY[i]]

    if NYYY[i] == 0:
        AAAA += [1]
    else :
        AAAA += [0]

    if NOut1[i] == 0:
        BBBB += [1]
    else :
        BBBB += [0]

    if i < predict_d:
        DDDD += [0]
    else :
        jj = 0
        for j in range(predict_d)[:-1] :
            jj = (1-0.1*jj) + 0.1 * NOut1[i-j]
        DDDD += [(jj*2 / predict_d)]
#        if jj > predict_d: 
#            DDDD += [1]
#        else :
#            DDDD += [0]

plt.plot(range(NOut1.shape[0]), CCCC)
plt.savefig('raw.png')

plt.plot(range(NOut1.shape[0]), DDDD)
plt.savefig('shift.png')

plt.plot(range(NOut1.shape[0]), BBBB)
plt.savefig('pred.png')


# In[15]:


#model.save('rnn_model_100_2_128_20.h5')

