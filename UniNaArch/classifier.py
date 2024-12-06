import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Visualize 10 random images from the dataset
plt.figure(figsize=(10, 2))
for i in range(10):
    index = np.random.randint(0, x_train.shape[0])  # Random index
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
    plt.title(y_train[index])
plt.show()

# Preprocess the data: Flatten images and normalize pixel values
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Define the FCNN model with an explicit Input layer
model = models.Sequential([
    layers.Input(shape=(28 * 28,)),  # Explicit Input layer
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define a callback to save the model at every epoch
checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/model_epoch_{epoch}.keras',
    save_freq=2*len(x_train)//32,
    save_weights_only=False,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model using the training data
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=2,
    callbacks=[checkpoint_callback, early_stopping]
)

# Evaluate the model using the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Plot the loss and validation loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

