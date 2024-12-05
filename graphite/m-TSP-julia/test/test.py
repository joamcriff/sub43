from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import HGSmTSP

import numpy as np
x = np.random.rand(100) * 1000
y = np.random.rand(100) * 1000
dist_mtx = np.sqrt((x[:,None] - x[None,:])**2 + (y[:,None] - y[None,:])**2)

coordinates = np.concatenate((x, y)).reshape(-1, 2)
HGSmTSP.solve_mTSP(3, dist_mtx, coordinates)