import tensorflow as tf
from tensorflow.keras import datasets, models
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST test dataset
(_, _), (x_test, y_test) = datasets.mnist.load_data()

# Select a random image from the test dataset
index = np.random.randint(0, x_test.shape[0])  # Random index
image = x_test[index]
label = y_test[index]

# Display the image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title(f'Actual Label: {label}')
plt.show()

# Preprocess the image: Flatten and normalize
image_processed = image.reshape(1, 28 * 28).astype('float32') / 255.0

# Load the trained model
model = tf.keras.models.load_model('checkpoints/model_epoch_9.keras')  # Update the filename if necessary

# Perform prediction
predictions = model.predict(image_processed)
predicted_label = np.argmax(predictions, axis=1)[0]

print(f'Predicted Label: {predicted_label}')
