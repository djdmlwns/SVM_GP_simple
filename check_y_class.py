from auxil.testfunc import Hosaki2d
import numpy as np
from auxil.awsauxil import send_datatos3
import boto3


s3_client = boto3.client('s3')

s3_client.download_file('testbucketjoonjae', 'input/y_array.npy', 'y_array.npy')
condition = lambda x: Hosaki2d(x) <= -0.5

# Data retrieve
X = np.load("new_x.npy")

y = np.load('y_array.npy')
# check classification of new_x and append

# check with benchmark function value
for _x in X:
    y = np.vstack([y, (1 if condition(_x) else -1)])

np.save('y_array.npy', y)

send_datatos3('y_array.npy', 'testbucketjoonjae', 'input/y_array.npy')