# %%
import boto3
import paramiko

# %%

client = boto3.client('ec2')
instDict=client.describe_instances(
    Filters = [
        {
            'Name' : 'instance-id',
            'Values' : ["i-0a6322874e9c2c4b8"]
        }
    ]
    # InstanceIds = ['i-0a6322874e9c2c4b8']
)

print(instDict)
# %%

s3_client = boto3.client('s3')
# Download private key file from secure S3 bucket
s3_client.download_file('testbucketjoonjae', 'keys/Joonjae_Ryu_2.pem', 'Joonjae_Ryu_2.pem')


k = paramiko.RSAKey.from_private_key_file("Joonjae_Ryu_2.pem")
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
# %%
