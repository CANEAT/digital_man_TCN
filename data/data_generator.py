##################################### load x and y####################

import os
import pickle
import numpy as np
import torch

def load_data(filepath):
    y_files = []
    x_files = []
    files = os.listdir(filepath)
    # files = filepath
    for file_names in files:
        if os.path.splitext(file_names)[1] == ".fm_y":
            y_files.append(file_names)
        elif os.path.splitext(file_names)[1] == ".wav_x":
            x_files.append(file_names)
    return x_files, y_files


def read_data(filename, filepath):
    data = []

    for filenames in filename:
        fp = open(filepath + "/" + filenames, 'rb')
        datamemory = pickle.load(fp)
        data.append(datamemory)
        fp.close()
    return data


def generator_data(filepath):
    x_files, y_files = load_data(filepath)
    dataX = read_data(x_files, filepath)
    # X = torch.from_numpy(np.array(dataX)).permute(1,0,2)
    X = np.array(dataX)
    dataY = read_data(y_files, filepath)
    Y = np.array(dataY)
    return X, Y

# filepath = 'E:\项目\deecamp\code\TCN-master\TCN-master\TCN\\adding_problem\output\\result'


def traindata(filepath):
    X, Y = generator_data(filepath)
    resultX, resultY = [], []
    for j in range(len(X)):
        x = X[j][:3600]
        x = np.vstack((zero_row(18), np.vstack((x, zero_row(12)))))
        resultX.append(x)
    for i in range(len(Y)):
        y = Y[i][:3600]
        resultY.append(y)
    return np.array(resultX), np.array(resultY)

def zero_row(num):
    return np.array([[0]*29]*num)

def numpy_to_tensor(x):
    return torch.unsqueeze(torch.from_numpy(x), dim=1)

# print(traindata(filepath))
# for i in range(5):
#     # print(traindata(filepath)[0])
#     print(torch.unsqueeze(torch.from_numpy(traindata(filepath)[0][i:i+2]), dim=1).shape)
#
#     print((traindata(filepath)[1]).shape)

# print(traindata(filepath)[0].shape)
# print(traindata(filepath)[0].shape)
# for i in range(len(traindata(filepath)[0])):
#     x = numpy_to_tensor(traindata(filepath)[0][i])
#     y = numpy_to_tensor(traindata(filepath)[1][i])
#     for j in range(20):
#         xtrain = x[j:j+20]
#         ytrain = y[j]
#     print(xtrain.shape)
#     print(ytrain.shape)
filepath = 'E:\项目\deecamp\data\DeeCamp_52\DeeCamp\data'
filepath1 = 'E:\项目\deecamp\code\Digital_man_TCN_pytorch1\Digital_man\output\\fmy_result'
print(traindata(filepath1)[1])
print(traindata(filepath)[1])