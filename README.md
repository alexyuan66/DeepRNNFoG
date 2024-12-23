Alexander M. Yuan's project: Using Deep Learning to Predict Freezing of Gait 
in Patients with Parkinson's Disease.

This project is supervised by Professor Shayok Chakraborty at Florida State 
University. The initial baseline code was provided by Professor Chakraborty and
produced the results in the paper:

Vishwas G. Torvi, Aditya Bhattacharya, and Shayok Chakraborty, "Deep Domain 
Adaptation to Predict Freezing of Gait in Patients with Parkinson's Disease," 
IEEE ICMLA 2018.

The baseline code no longer works with the current version of Python and 
TensorFlow. This repository contains the updated programs with the 
following functionality that was added:

1. Updated the code to use Keras instead of Tensorflow to build the RNN network.
2. Expanded the type of recurrent neural network from LSTM to LSTM, SimpleRNN, GRU, and the directional versions of these networks.
3. Supported various command options to set neural network hyperparameters and other training parameters (af, n_a, Tx, predict_d, Tstep, Tcutoff, Ssetp, batch_size, epochs, TFILE, TFILE2, dropout rate, number of RNN layers, etc). See new_gRNN_paper_2layer_v6.py for details.
4. Supported both regular training and two transfer learning methods.  

