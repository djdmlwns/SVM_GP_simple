# %%
from sklearn import svm
import numpy as np
from core.asopt import MyGPR, ActiveSampling
from auxil.testfunc import Hosaki2d
from auxil.awsauxil import send_datatos3
import boto3

#%%
s3_client = boto3.client('s3')

# Download private key file from secure S3 bucket
s3_client.download_file('testbucketjoonjae', 'input/X_array.npy', 'X_array.npy')
s3_client.download_file('testbucketjoonjae', 'input/y_array.npy', 'y_array.npy')

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

#%%

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
new_x = AS.find()

# %%

np.save('X_array.npy', AS.X)
np.save('new_x.npy', new_x)

send_datatos3('X_array.npy', 'testbucketjoonjae', 'input/X_array.npy')
send_datatos3('new_x.npy', 'testbucketjoonjae', 'input/new_x_array.npy')

print('Added:', new_x)

#%%
