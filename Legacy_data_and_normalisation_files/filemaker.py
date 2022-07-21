import numpy as np

norms = []
objids = []
for i in range(10):
    np.savez(f"/cosma5/data/durham/dc-will10/exdatanorms{i+1}.npz", norms = norms, objids = objids)