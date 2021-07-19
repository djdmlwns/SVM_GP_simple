import boto3
import logging
from botocore.exceptions import ClientError


def send_datatos3(file_name, bucket, object_name = None):
    # send text file to server for simulation
    # adequate formating should be known in the future
    if object_name is None:
        object_name = file_name
    
    s3_client = boto3.client('s3')

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False

    return True
    # return True
