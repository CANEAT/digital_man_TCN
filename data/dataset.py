from .data_generator import generator_data
import torch
import numpy as np

def TrainData(root):
    X, Y = generator_data(root)
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

def ValData(root):
    X, Y = generator_data(root)
    resultX, resultY = [], []
    for j in range(len(X)):
        x = X[j][:3600]
        x = np.vstack((zero_row(18), np.vstack((x, zero_row(12)))))
        resultX.append(x)
    for i in range(len(Y)):
        y = Y[i][:3600]
        resultY.append(y)
    return np.array(resultX), np.array(resultY)

def TestData(root):
    Data = generator_data(root)
    X_val = torch.from_numpy(Data[0]).permute(1, 0, 2)
    return X_val
