import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

L=23.5
l=8.6

inouts = np.load("inouts.npy")
rel_diffs=np.load("rel_diffs.npy")

x = inouts[:, 0]         # ∈ [0, L]
y = inouts[:, 1]         # ∈ [0, l]
z = rel_diffs.ravel()    # relative errors

# define a regular grid over the room
nx, ny = 200, 200
xi = np.linspace(0, L, nx)
yi = np.linspace(0, l, ny)
xi, yi = np.meshgrid(xi, yi)

# interpolate scattered data onto the grid
zi = griddata((x, y), z, (xi, yi), method='cubic')

# plot
plt.figure()
plt.imshow(zi, 
           origin='lower', 
           extent=(0, L, 0, l), 
           aspect='auto')
plt.colorbar(label='Relative Error')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Heat-map of Relative Error across Room')
plt.show()
