import matplotlib.pyplot as plt
import numpy as np

inouts = np.load("inouts.npy")

xx = np.linspace(2.0,2.2,len(inouts))
plt.plot(xx,xx)
plt.plot(inouts[:,2], inouts[:,3], linestyle="", marker="o")
plt.show()
