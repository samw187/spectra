import numpy as np

norms = []
objids = []
for i in range(10):
    np.savez(f"/cosma/home/durham/dc-will10/datanorms{i+1}.npz", norms = norms, objids = objids)