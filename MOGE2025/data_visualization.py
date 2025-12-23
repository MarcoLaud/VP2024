import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('VPM2025_simulations.csv', delimiter=',', skip_header=1)

x, y, z = data[:, 0], data[:, 1], data[:, 2]

# plt.figure()

# # Filled contour on scattered (x,y) with z as color (Interpolate!)
# cntr = plt.tricontourf(x, y, z, levels=50)

# # Show the sample locations
# plt.scatter(x, y, s=15, edgecolor='none')

# plt.colorbar(cntr, label='T60')   # or 'Relative error' if that's what z is
# plt.xlim(0, 23.5)
# plt.ylim(0, 8.6)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('Colour map over room from scattered samples')
# plt.tight_layout()
# plt.show()

# No interpolation:
plt.figure()
sc = plt.scatter(x, y, c=z, s=120, marker='s', edgecolors='k')
plt.colorbar(sc, label='Value')
plt.xlim(0, 23.5); plt.ylim(0, 8.6); plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.title('Sample values (no interpolation)')
plt.show()
