from auxil.testfunc import Hosaki2d
import numpy as np
from auxil.awsauxil import send_datatos3
import boto3
import sys
import json

condition = lambda x: Hosaki2d(x) <= -0.5

# Data retrieve
x_str = sys.argv[1] # 

# Convert string to numpy array
x_array = np.fromstring(x_str[1:-1], dtype = np.float, sep = ',')

# check with benchmark function value
y = 1 if condition(x_array) else -1

with open('y.txt', 'w') as f:
    json.dump(y, f)

send_datatos3('y.txt', 'testbucketjoonjae', 'input/y.txt')

s3_client = boto3.client('s3')

