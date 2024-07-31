import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the problem parameters
L = 1.0      # Length of the string
c = 1.0      # Wave speed
A = 0.1      # Amplitude of the initial displacement
T = 1.0      # Total time duration

# Define the neural network architecture
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(75, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(75, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(75, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(75, activation='tanh')
        self.dense5 = tf.keras.layers.Dense(75, activation='tanh')
        self.dense6 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.dense6(x)

# Create an instance of the PINN model
model = PINN()

# Define the analytical solution for comparison
def analytical_solution(x, t, L, c, A):
    omega = np.pi * c / L
    return A * np.sin(np.pi * x / L) * np.cos(omega * t)

# Define the loss function components
def loss_pde(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)
    u_tt = tape.gradient(u_t, t)
    del tape
    return tf.reduce_mean((u_tt - c**2 * u_xx)**2)

def loss_bc(x, t):
    u_pred = model(tf.concat([x, t], axis=1))
    return tf.reduce_mean(u_pred**2)

def loss_ic(x, t, u_initial):
    u_pred = model(tf.concat([x, t], axis=1))
    return tf.reduce_mean((u_pred - u_initial)**2)

# Generate training data
n_train = 1000
x_train = np.random.rand(n_train, 1) * L
t_train = np.random.rand(n_train, 1) * T

# Initial condition at t=0
x_ic = np.random.rand(n_train, 1) * L
t_ic = np.zeros((n_train, 1))
u_initial = A * np.sin(np.pi * x_ic / L)

# Boundary condition at x=0 and x=L
t_bc = np.random.rand(n_train, 1) * T
x_bc_0 = np.zeros((n_train, 1))
x_bc_L = L * np.ones((n_train, 1))

# Convert data to TensorFlow tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
x_ic = tf.convert_to_tensor(x_ic, dtype=tf.float32)
t_ic = tf.convert_to_tensor(t_ic, dtype=tf.float32)
u_initial = tf.convert_to_tensor(u_initial, dtype=tf.float32)
x_bc_0 = tf.convert_to_tensor(x_bc_0, dtype=tf.float32)
x_bc_L = tf.convert_to_tensor(x_bc_L, dtype=tf.float32)
t_bc = tf.convert_to_tensor(t_bc, dtype=tf.float32)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Checkpointing
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "pinn_ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore the latest checkpoint if available
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("No checkpoint found. Starting training from scratch.")

# Training loop
epochs = 200000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_pde(x_train, t_train) + \
               loss_ic(x_ic, t_ic, u_initial) + \
               loss_bc(x_bc_0, t_bc) + \
               loss_bc(x_bc_L, t_bc)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
        manager.save()
        print("Checkpoint saved at epoch {}".format(epoch))

