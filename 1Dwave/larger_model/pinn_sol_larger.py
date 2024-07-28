import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
L = 1.0      # Length of the string
c = 1.0      # Wave speed
A = 0.1      # Amplitude of the initial displacement
T = 1.0      # Total time duration

# Define the neural network architecture (same as the training model)
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

# Restore the latest checkpoint
checkpoint_dir = './checkpoints'

checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print(f"Model restored from checkpoint: {manager.latest_checkpoint}")
else:
    print("No checkpoint found. Exiting.")
    exit()

# Define grid and time points
nx = 100
x_vals = np.linspace(0, L, nx)
times = [0.0, 0.25, 0.5, 0.75, 1.0]

# Compute the solutions and visualize the results
L2_norm_diffs = []

for t in times:
    t_vals = t * np.ones_like(x_vals)
    inputs = np.column_stack((x_vals, t_vals))

    # Ensure inputs are in the correct shape and dtype
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    u_pinn = model(inputs_tf).numpy().flatten()  # Ensure model output is flat
    u_analytical = analytical_solution(x_vals, t, L, c, A).flatten()

    # Compute the L2 norm difference
    numerator = np.sum((u_pinn - u_analytical)**2)
    denominator = np.sum(u_pinn**2)
    L2_norm_diff = np.sqrt(numerator / denominator)
    L2_norm_diffs.append(L2_norm_diff)

    # Plotting
    plt.figure()
    plt.plot(x_vals, u_analytical, label='Analytical')
    plt.plot(x_vals, u_pinn, label='PINN', linestyle='dashed')
    plt.title(f"Solution Comparison at t = {t}")
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(-0.1, 0.1)
    plt.show()

# Print L2 norm differences
for t, diff in zip(times, L2_norm_diffs):
    print(f"L2 norm difference at t={t}: {diff}")
