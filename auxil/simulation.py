# %%
import numpy as np
from auxil.awsauxil import send_datatos3
import boto3
import logging

class Simulation():
    '''
    This needs to be modified 
    '''
    def __init__(self, x, **kwargs):
        self.result = None
        self.x = x
        with open('input.txt', 'w') as f:
            f.write(str(x))


    def run(self):
        # use data for simulation
#        data = self.x 
        send_datatos3('input.txt', 'testbucketjoonjae', 'input.txt')
        
        result_text = self.retrieve_result()
        result = self.postprocessing_result(result_text)        

        self.result = result


    def retrieve_result(self):
        # read result text file from server
        with open('output.txt', 'r') as f:
            lines = f.read()
        return lines


    def postprocessing_result(self, result_text):
        # postprocess text to be used in Python
        processed_result = result_text
        return processed_result 