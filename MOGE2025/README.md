# MOGE 2025
## A simple neural network model to predict the reverberation time in a room

'Regression_01.py' implements a neural network to predict the reverberation time (500 Hz) in a room of the école nationale supérieure d’architecture de paris-belleville. The model takes in input the position of the center of an absorption panel and yields the reverberation time.

The training dataset is in "data_fixed.csv", and is based on 20 simulations from Pachyderm (Rhino). The positions of the panel are obtained via a Latine Lattice Sampling as defined in "LLH.py".

The files with extension ".gh" and ".3dm" contain the Grasshopper 3D implementation of the neural network (in PUG).

