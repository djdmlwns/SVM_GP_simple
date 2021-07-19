import boto3
import paramiko
def lambda_handler(event, context):

    s3_client = boto3.client('s3')

    # Download private key file from secure S3 bucket
    s3_client.download_file('testbucketjoonjae', 'keys/Joonjae_Ryu_2.pem', '/tmp/Joonjae_Ryu_2.pem')

    k = paramiko.RSAKey.from_private_key_file("/tmp/Joonjae_Ryu_2.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    host = 'ec2-3-136-25-127.us-east-2.compute.amazonaws.com'
    print("Connecting to " + host)
    c.connect( hostname = host, username = "ec2-user", pkey = k )
    print("Connected to " + host)

    commands = [
            "conda activate surrogate; cd gittest1/SVM_GP_simple; python initial_code.py",
    ]

    for command in commands:
        print("Executing {}".format(command))
        stdin, stdout, stderr = c.exec_command(command)
        print(stdout.read())
        print(stderr.read())

    return 
    {
        'message' : "Script execution completed. See Cloud logs for output"
    }

