import boto3
from auxil.testfunc import Hosaki2d
import numpy as np
from auxil.awsauxil import senddatatos3

condition = lambda x: Hosaki2d(x) <= -0.5

s3_client = boto3.client('s3')

# Data retrieve
X = np.load("X_initial_array.npy")

y = []
# check classification of new_x and append

# check with benchmark function value
for _x in X:
    y.append(1 if condition(_x) else -1)

y_array = np.array(y)
np.save('y_initial_array.npy', y_array)


