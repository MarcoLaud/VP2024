# Regression model to estimate the reverbaration time of a 2D shoebox.
# Fully connected neural network model.
# Pre-processing: normalization [0,1]
# Dataset: 30 simulations (x,y) -> T_rev (500 Hz)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import random
from tensorflow.keras.initializers import GlorotUniform, Zeros
from sklearn.model_selection import LeaveOneOut

def build_model(seed):
    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,), kernel_initializer=GlorotUniform(seed=seed), bias_initializer=Zeros()),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=seed), bias_initializer=Zeros()),
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=GlorotUniform(seed=seed), bias_initializer=Zeros()),
        tf.keras.layers.Dense(1, activation='relu', kernel_initializer=GlorotUniform(seed=seed), bias_initializer=Zeros())])

    #------------
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse')
    return model

#--------------
# Data loading
data = np.genfromtxt('dati_MOGE_30.txt', delimiter=',', skip_header=1)

#--------------
# Building training, validation, and test datasets
X = data[:,:2]                # input data shape: (20,2)
y = data[:,-1].reshape(-1,1)  # output data shape: (20,1)

#-------------
# Leave-one-case-out study:
loo = LeaveOneOut()
errs = np.zeros(30)
jj=0  # loop idx
for train_idx, test_idx in loo.split(X):
    tf.keras.backend.clear_session()  # clear the session

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    #-------------
    # Data normalization [0,1]
    # compute mins & maxs on the training set (inputs)
    x_min = X_train.min(axis=0)   # shape (2,)
    x_max = X_train.max(axis=0)   # shape (2,)

    # scale into [0,1]
    X_train_01 = (X_train - x_min) / (x_max - x_min)
    X_test_01  = (X_test  - x_min) / (x_max - x_min)

    # compute mins & maxs on the training set (outputs)
    y_min = y_train.min(axis=0)   # shape (1,)
    y_max = y_train.max(axis=0)   # shape (1,)

    # scale into [0,1]
    y_train_01 = (y_train - y_min) / (y_max - y_min)
    y_test_01  = (y_test  - y_min) / (y_max - y_min)

    #------------
    # Choose a seed
    seed = 69  #LMAO

    # 2) Seed Python, NumPy, and TensorFlow global RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Build model:
    model = build_model(seed)

    history = model.fit(
        X_train_01, y_train_01,
        epochs=50,
        verbose=0)

    #-----------
    test_mse_n = model.evaluate(X_test_01, y_test_01, verbose=0)
    print("------------------------------------------------------------")
    print("CASE {}: Normalized test MSE: {:.4f}".format(jj, test_mse_n))

    # get normalized prediction and invert scaling
    y_pred_n = model.predict(X_test_01)
    y_pred = y_pred_n * (y_max - y_min) + y_min

    print("TRUE:      ", y_test.flatten())
    print("PREDICTED: ", y_pred.flatten())
    rel_diff = np.abs(y_test.flatten() - y_pred.flatten()) / y_test.flatten() * 100
    print("RELATIVE DIFFERENCE: {}%".format(rel_diff))
    print("------------------------------------------------------------")
    errs[jj] = rel_diff[0]
    jj +=1

print("############################")
print("MEAN RELATIVE ERROR: {:.4f}".format(np.mean(errs)))
print("############################")

# Save errors:
np.save("rel_diffs.npy", errs)
