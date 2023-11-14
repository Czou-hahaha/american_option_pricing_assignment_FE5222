import time
import numpy as np
from QdPlus import QdInitial

r = q = 0.04
S = 100
K = 100
T = 3
t = 0
n = 10
vol = 0.2
tau_vec = np.array(range(0,n+1))/(T-t)
qd_solver = QdInitial(r,q,S,K,vol,tau_vec)
tic = time.time()
boundaries = qd_solver.init_price()
print(time.time()-tic)
print(boundaries)
