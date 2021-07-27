# %%
import numpy as np
import boto3
import os
import datetime
import time
import sys
import json

class Simulation():
    '''
    Run simulation to get the classification result y
    '''
    def __init__(self, x, **kwargs):
        self.result = None
        self.x = x
        self.jobqueue = 'SVM_Opt_Test'
        self.jobdefinition = 'SVM_Test_Batch_RefInput'

        if len(x.shape) == 1:
            self.arraysize = 1
        else:
            self.arraysize = x.shape[0]

        # data folder to save input
        folder = "data"        
        self.localfoldername = folder

        self.inputname = []
        for array in range(self.arraysize):
            self.inputname.append('input' + str(array) + '.txt')

        d = datetime.datetime.now()        
        self.timestamp = str(d.year) + str(d.month) + str(d.day) + str(d.hour) + str(d.minute) + str(d.second)
        self.jobid = None

        os.chdir(".")
        
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for array in range(self.arraysize):
            with open(self.localfoldername + '/' + self.inputname[array], 'w') as f:
                print('self array size', self.arraysize, 'iteration array', array)
                f.write(str(np.atleast_2d(x)[array, :]))

    def run(self):
        # use data for simulation
#        data = self.x 
        for array in range(self.arraysize):
            self.send_data('testbucketjoonjae', self.localfoldername + '/' + self.inputname[array], 'input' + '/' + self.inputname[array])
        jobname = 'SVM' + self.timestamp

        # submit batch job
        self.submit_batch(jobname, self.jobqueue, self.jobdefinition, arraysize = self.arraysize)
        print("Job is submitted")

        # check batch status
        if self.checkbatchstatus() == True:
            # pick random class for dummy test and upload to bucket for testing
            # should be removed for future use
            ####################################################
            for array in range(self.arraysize):
                y_data = np.random.randint(2)
                data = {}
                data['status'] = y_data
                with open('output' + str(array) + '.json', 'w') as f:
                    json.dump(data, f)
                
                resource = boto3.resource('s3')
                resource.meta.client.upload_file('output' + str(array) + '.json', 'testbucketjoonjae', 'simulation_output/' + 'output' + str(array) + '.json')
            ######################################################

            # read true result from s3 bucket
            y_list = self.read_resultfroms3('testbucketjoonjae')                
            print('y is obtained')
        else:
            print("Simulation fails, so algorithm is terminated")
            sys.exit()     

        filter = lambda x: -1 if x == 0 else x
        y_list = list(map(filter, y_list))

        return y_list

    def send_data(self, bucketname, localfilename, cloudfilename = 'input/input.txt'):
        # send text file to server for simulation
        # adequate formating should be known in the future
        resource = boto3.resource('s3')
        resource.meta.client.upload_file(localfilename, bucketname, cloudfilename)
        print('Data is sent')


    def checkbatchstatus(self, max_retry = 10):
        client = boto3.client('batch')        

        wait = True
        success = False
        trial = 0

        while wait and trial <= max_retry :
            jobinfo = client.describe_jobs(jobs = [self.jobid])['jobs'][0]                        
            job_status = jobinfo['status']

            if job_status == 'SUCCEEDED':
                print('Job is done successfully')
                success = True
                break

            elif job_status == 'FAILED':
                break

            else:
                print('Current job status:', job_status)
                time.sleep(10)
                trial += 1

        return success

    def read_resultfroms3(self, bucketname):
        
        y_list = []
        # download data from s3
        s3 = boto3.resource('s3')

        for array in range(self.arraysize):
            outputfilename = 'output' + str(array) + '.json'
            s3.Bucket(bucketname).download_file('simulation_output/' + outputfilename, outputfilename)
            with open(outputfilename, 'r') as f:
                result = json.load(f)

            y_list.append(result['status'])

        print('json file downloaded')

        # read result text file from server
        return y_list

    def postprocessing_result(self, result_text):
        # postprocess text to be used in Python
        processed_result = result_text
        return processed_result 


    def submit_batch(self, jobname, jobqueue, jobdefinition, arraysize = 1):
        client = boto3.client('batch')

        if arraysize == 1 or arraysize == 0 :
            response = client.submit_job(
                jobName = jobname,
                jobQueue = jobqueue,
                jobDefinition = jobdefinition,

    #            containerOverrides = containeroverrides,
    #            parameters = parameters
            )
        else:
            response = client.submit_job(
                jobName = jobname,
                jobQueue = jobqueue,
                jobDefinition = jobdefinition,
                arrayProperties = {'size' : arraysize}
    #            containerOverrides = containeroverrides,
    #            parameters = parameters
            )

        jobID = response['jobId']
        self.jobid = jobID
        return jobID