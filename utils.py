import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px # to plot the time series plot
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
# import tensorflow as tf 
import torch
import torch.nn as nn
from torch.autograd import Variable
import imageio
import natsort
import os

# This is the sliding window for the LSTM
def sliding_windows(data, seq_length):
    x = []
    predicted_ls = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        predicted = data[i+seq_length]
        x.append(_x)
        predicted_ls.append(predicted)

    return np.array(x), np.array(predicted_ls)

# This method is to convert X and Y values into tensors and reshape them
def conv_and_reshape(x, y):

    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    if len(X.shape) == 1:
        X = torch.reshape(X, (X.shape[0],1))
        Y = torch.reshape(Y, (Y.shape[0],1))
    elif len(X.shape) == 2:
        X = torch.reshape(X, (X.shape[0],X.shape[1],1))
        Y = torch.reshape(Y, (Y.shape[0],Y.shape[1],1))
    return X, Y

# This method is to get data for X and Y 
def get_data(df, inst_id, seq_length, cols):
    target_df = df[df.instance_id == inst_id]

    df_cols_training = target_df[cols]

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_cols_training = scaler.fit_transform(df_cols_training)

    df_cols_training_x = df_cols_training[:,0]
    df_cols_training_y = df_cols_training[:,1]

    x, expected_x = sliding_windows(df_cols_training_x, seq_length)
    y, expected_y = sliding_windows(df_cols_training_y, seq_length)

    dataX, dataY = conv_and_reshape(x,y)
    expected_x, expected_y = conv_and_reshape(expected_x, expected_y)

    return dataX, dataY, expected_x, expected_y

# This method is to get data for X,Y,Z and heading angle 
def get_all_data(df, inst_id, seq_length, cols):
    target_df = df[df.instance_id == inst_id]

    df_cols_training = target_df[cols]

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_cols_training = scaler.fit_transform(df_cols_training)

    df_cols_training_x = df_cols_training[:,0]
    df_cols_training_y = df_cols_training[:,1]
    df_cols_training_z = df_cols_training[:,2]
    df_cols_training_r = df_cols_training[:,3]

    x, expected_x = sliding_windows(df_cols_training_x, seq_length)
    y, expected_y = sliding_windows(df_cols_training_y, seq_length)
    z, expected_z = sliding_windows(df_cols_training_z, seq_length)
    r, expected_r = sliding_windows(df_cols_training_r, seq_length)

    dataX, dataY = conv_and_reshape(x,y)
    dataZ, dataR = conv_and_reshape(z,r)
    expected_x, expected_y = conv_and_reshape(expected_x, expected_y)
    expected_z, expected_r = conv_and_reshape(expected_z, expected_r)

    return dataX, dataY, expected_x, expected_y, dataZ, dataR, expected_z, expected_r

def conv_to_gif():
    
    images = []
    for filename in natsort.natsorted(os.listdir("./pictures/")):
        print(filename)
        for i in range(4):
            images.append(imageio.imread("./pictures/"+filename))
    imageio.mimsave('./gif/path.gif', images)

    for filename in natsort.natsorted(os.listdir("./pictures/")):
        os.remove("./pictures/"+filename)