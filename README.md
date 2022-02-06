Alexander M. Yuan's project: Using Deep Learning to Predict Freezing of Gait 
in Patients with Parkinson's Disease.

This project is supervised by Professor Shayok Chakraborty at Florida State 
University. The initial baseline code is provided by Professor Chakraborty that
produce the results in paper:

Vishwas G. Torvi, Aditya Bhattacharya, and Shayok Chakraborty, "Deep Domain 
Adaptation to Predict Freezing of Gait in Patients with Parkinson's Disease," 
IEEE ICMLA 2018.

The baseline code no longer works in the current version of python and 
tensorflow. This repository contains many versions of updated programs with the 
following main added functionality:

1. Update the code to use Keras instead of tensorflow to build the RNN network.
2. Expand the type of recurrent neural network from LSTM only to LSTM, SimpleRNN, GRU, as well as the directional versions of these networks.
3. Support many command options to set neural entwork hyperparamemters and other training parameters (af, n_a, Tx, predict_d, Tstep, Tcutoff, Ssetp, batch_size, epoches, TFILE, TFILE2, dropout rate, number of RNN layers, etc). See new_gRNN_paper_2layer_v6.py for details).
4. Support both regular training and two transfer learning methods.  

