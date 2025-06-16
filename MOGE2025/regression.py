import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

#--------------
# fixing commas into dots and data loading
with open("./dati_MOGE.csv", "r") as fin:
    text = fin.read().replace(",", ".")
with open("data_fixed.csv", "w") as fout:
    fout.write(text)

data = np.genfromtxt('data_fixed.csv', delimiter=';', skip_header=1)

#--------------
# Building training, validation, and test datasets
X = data[:,:2]                # input data shape: (20,2)
y = data[:,-1].reshape(-1,1)  # output data shape: (20,1)

# Here we use set validation and test samples:
val_idx  = 3
test_idx = 2

X_val, y_val   = X[val_idx:val_idx+1],   y[val_idx:val_idx+1]
X_test, y_test = X[test_idx:test_idx+1], y[test_idx:test_idx+1]

# Training indices are all the rest:
train_idx = [i for i in range(len(X)) if i not in (val_idx, test_idx)]
X_train, y_train = X[train_idx], y[train_idx]

#-------------
# Data normalization
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

# transform all sets
X_train_n = x_scaler.transform(X_train)
X_val_n   = x_scaler.transform(X_val)
X_test_n  = x_scaler.transform(X_test)

y_train_n = y_scaler.transform(y_train)
y_val_n   = y_scaler.transform(y_val)

#------------
# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

#------------
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse')

#------------
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train_n, y_train_n,
    validation_data=(X_val_n, y_val_n),
    epochs=500,
    callbacks=[es],
    verbose=0
)

#-----------
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
epochs     = range(1, len(train_loss) + 1)

# Plot both training and validation curves
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

#-----------
test_mse_n = model.evaluate(X_test_n, y_scaler.transform(y_test), verbose=0)
print(f"Normalized test MSE: {test_mse_n:.4f}")

# get normalized prediction and invert scaling
y_pred_n = model.predict(X_test_n)
y_pred   = y_scaler.inverse_transform(y_pred_n)

print("TRUE:      ", y_test.flatten())
print("PREDICTED: ", y_pred.flatten())
rel_diff = np.abs(y_test.flatten() - y_pred.flatten()) / y_test.flatten() * 100
print("RELATIVE DIFFERENCE: {}%".format(rel_diff))
