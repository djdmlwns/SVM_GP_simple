# %%
from sklearn import svm
import numpy as np
from core.asopt import MyGPR, ActiveSampling

#%%
# Data retrieve
X = np.load("X_array.npy")
print('previous:', X)

y = np.load("y_array.npy")
y = y.tolist()
dim = X.shape[1]
condition = lambda x: Hosaki2d(x) <= -0.5

# Variable bound
bounds = []
for i in range(dim):
    bounds.append((0.0, 1.0))   # [0, 1]^n. For simulation, normalization of X is needed

# Start Active Learning/Sampling algorithm 

# initialize SVM
SVM_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42) 

# initialize GP regressor
GP_regressor = MyGPR(normalize_y = True, n_restarts_optimizer = 0, alpha = 1e-7) 

# initialize Active sampling algorithm class 
AS = ActiveSampling(X, y, SVM_classifier, GP_regressor, bounds, 
                    case = 'benchmark', C1=1, condition = condition, p_check = 0.0, threshold = 1, 
                    C1_schedule='None', acq_type = 'f1', cal_norm = True, 
                    n_optimization=5)

# training start
AS.find()

np.save('X_array.npy', AS.X)
np.save('y_array.npy', np.array(AS.y))

print('new:', AS.X)
