#!/usr/bin/env python
# coding: utf-8
# Alexander M. Yuan
# May 15, 2020
#
# This change the LSTM code to output in every cell instead of the last cell
# The code is modified from Al Created on May 15 2020
#
# This is the jupyter program that is modified from Prof. Chakraborty's RNN for FoG detection for PD patience. Trying to reproduce the results in paper: 

# Vishwas G. Torvi, Aditya Bhattacharya, and Shayok Chakraborty, "Deep Domain Adaptation to Predict Freezing of Gait in Patients with Parkinson's Disease," IEEE ICMLA 2018.

#The baseline code is provided by Prof. Chakraborty. The code no longer works in the current version of python and tensorflow. Rewrote the code to generate X and Y arrays for training set and testing set, use Keras instead of tensorflow to build the RNN network.
# cell to import packages: numpy, matplotlib, tensorflow, keras, sklearn.metrics

# In[1]:


import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  

from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import SimpleRNN, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from sklearn import metrics
from tqdm.keras import TqdmCallback


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

# file directory, 
#train data and test set are all in the directory
# 2/3 data for training, 1/3 data for testing
TRAIN = '../../dataset_fog_release/dataset/Train/'
#TRAIN = 'D:/FoG/dataset_fog_release/dataset/Train/'

# parameters
# prediction and RNN parameter, related to how to arrange the data
predict_d = 300  # how many samples in the forward FOG prediction
Tx = 50 # number of time steps in the RNN
Tstep = 1 # RNN examines data every Tstep entries

Tcutoff = 6000000 # Examine at most Tcutoff*0.015 seconds before FoG
Sstep = 1 # sample step is Sstep

n_a=50 # RNN internal size


# ## New code to load data Data Loading
# the output needs to change have 

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
def gen_batch_one_episode(data, beg, end, shuffle=1):
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
    
    for i in [n for n in range(beg, end-Tx*Tstep, Sstep)]:
        t = []
        tt = []
        for j in range(Tx):
            t += [data[i+j*Tstep][:9]]

        if i+j*Tstep + predict_d > end-1: 
            if data[end-1][9] == '2':
                Y += [[0, 1]]
            else :
                Y += [[1, 0]]
        elif data[i+j*Tstep+predict_d][9] == '2':
            Y += [[0, 1]]
        else :
            Y += [[1, 0]]
        X += [t]

    if shuffle == 0:
        return X, Y
    
    X1, Y1 = shuffle_list(X, Y)
    
    #for i in range(len(X)):
    #    if Y[i] == [0, 1]:
    #        XX += [X[i]]
    #        YY += [Y[i]]
    return X1, Y1

# generate X, Y for all episode
def gen_batch(data, shuffle=1):
    global e_count
    X = []
    Y = []
    X1 = []
    Y1 = []
    beg = 0
    end = beg
    print('len=', len(data))
    while beg < len(data):
        end = beg
        while end < len(data) and data[end][9] != '2':
            # print(end, data[end], '**', data[end][9])
            end = end + 1
        while end < len(data) and data[end][9] == '2':
            end = end + 1
        if data[end-1][9] == '2': 
            e_count = e_count+1
            
        X1, Y1 = gen_batch_one_episode(data, beg, end, shuffle=shuffle)
        X += X1
        Y += Y1
        beg = end
    return X, Y
        
def load_data(file_paths):
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
            
            train += X_signals[:2*int(len(X_signals)/3)]
            test += X_signals[2*int(len(X_signals)/3):]
            #train += X_signals[len(X_signals)-120:len(X_signals)-60]
            #test += X_signals[len(X_signals)-60:]
    
    train = np.array(train)
    train_X, train_Y = gen_batch(train, shuffle=0)
    print(e_count, ' episodes in training data')
    e_count = 0
    test_X, test_Y = gen_batch(test, shuffle=0)
    print(e_count, ' episodes in testing data')
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_data(TRAIN)
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
for i in range (train_X.shape[0]) :
    if train_Y[i][0] == 0: 
        c += 1

print('train_Y count =', c)

c = 0;
for i in range (test_X.shape[0]) :
    if train_Y[i][0] == 0: 
        c += 1

print('test_Y count =', c) 

#exit(0)

#sample = open('output.txt', 'w')
#np.set_printoptions(threshold=sys.maxsize)
#print(train_Y, file=sample)
#sample.close()


# # Build the RNN
#    Code is copied and modified from the coursera course


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

    model = Sequential()
    model.add(SimpleRNN(n_a, activation='relu', return_sequences=True, input_shape=(Tx, n_values)))
    model.add(SimpleRNN(n_a, activation='relu'))
    model.add(densor)
    
    
    ## Step 1: Create empty list to append the outputs while you iterate
    #outputs = []
    #
    ## Step 2: Loop
    #
    #for t in range(Tx):
    #    
    #    x = Lambda(lambda z: z[:, t, :])(X)
    #    x = reshapor(x)
    #    a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
    #    out=densor(a)
    #    outputs.append(out)
#
 #   #print(outputs)
  
#  # Step 3: Create model instance
 #   model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


# In[8]:


model = mymodel(Tx=Tx, n_a=n_a, n_values = 9)
model.summary()

# In[9]:


opt = Adam(lr=0.01, beta_1 = 0.9, beta_2=0.999, decay = 0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# # initialize hidden state and cell state

# In[10]:


XXX=train_X
YYY = train_Y

a0 = np.zeros((len(XXX), n_a))
c0= np.zeros((len(XXX), n_a))
print(YYY.shape)


# In[11]:

#model.fit(XXX, list(YYY.transpose((1, 0, 2))), batch_size=128,
#          epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=0)])

model.fit(XXX, YYY, batch_size=1000, epochs=5) # callbacks=[TqdmCallback(verbose=0)])

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
    if YYY[i][0] == 0:
        c0 += 1
    if Out1[i][0] != YYY[i][0]:
        if Out1[i][0] == 0:
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
NOut = model.predict(NXXX)

NOut1 = np.around(NOut)
print(NOut1.shape[0])
print(NOut1.shape)

Nc0=0
Nc1=0
Nc2=0
for i in range(NOut1.shape[0]):
    if NYYY[i][0] == 0:
        Nc0 += 1
    if NOut1[i][0] != NYYY[i][0]:
        if NOut1[i][0] == 0:
            Nc1 += 1
        else : 
            Nc2 += 1
            
print('Nc0 = ', Nc0)
print('Nc1 = ', Nc1)
print('Nc2 = ', Nc2)


# In[15]:


model.save('rnn_model_100_2_128_20.h5')

