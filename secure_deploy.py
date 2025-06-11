"""
secure_deploy.py

This script provisions secure AWS resources for MedGPT deployment using boto3.
It demonstrates best practices for HIPAA/GDPR/SOC2 compliance, including:
- Encrypted S3 bucket
- IAM role with least privilege
- CloudTrail logging
- Security group for EC2

Before running, configure your AWS credentials and region.
"""
import boto3
import json
from sagemaker import Session, get_execution_role
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
import time

# Create encrypted S3 bucket
def create_encrypted_bucket(bucket_name):
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=bucket_name)
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules': [{
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'AES256'
                }
            }]
        }
    )
    print(f"Encrypted S3 bucket '{bucket_name}' created.")

# Create IAM role with least privilege
def create_iam_role(role_name):
    iam = boto3.client('iam')
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy)
    )
    print(f"IAM role '{role_name}' created.")
    return role

# Enable CloudTrail logging
def enable_cloudtrail(trail_name, bucket_name):
    ct = boto3.client('cloudtrail')
    ct.create_trail(Name=trail_name, S3BucketName=bucket_name)
    ct.start_logging(Name=trail_name)
    print(f"CloudTrail '{trail_name}' enabled.")

# Create security group for EC2
def create_security_group(group_name, vpc_id):
    ec2 = boto3.client('ec2')
    sg = ec2.create_security_group(
        GroupName=group_name,
        Description='Secure group for MedGPT',
        VpcId=vpc_id
    )
    print(f"Security group '{group_name}' created.")
    return sg

def create_fsx_lustre(fsx_name, s3_import_path, vpc_id, subnet_id, security_group_id):
    fsx = boto3.client('fsx')
    response = fsx.create_file_system(
        FileSystemType='LUSTRE',
        StorageCapacity=1200,  # Minimum for persistent deployment
        SubnetIds=[subnet_id],
        SecurityGroupIds=[security_group_id],
        LustreConfiguration={
            'DeploymentType': 'PERSISTENT_1',
            'PerUnitStorageThroughput': 200,
            'ImportPath': s3_import_path,
            'ExportPath': s3_import_path,
            'DataCompressionType': 'LZ4',
            'AutomaticBackupRetentionDays': 7,
            'CopyTagsToBackups': True,
            'DriveCacheType': 'READ'
        },
        Tags=[{'Key': 'Name', 'Value': fsx_name}]
    )
    fsx_id = response['FileSystem']['FileSystemId']
    print(f"FSx for Lustre '{fsx_id}' created and linked to S3.")
    return fsx_id

def create_sagemaker_pipeline(s3_input, s3_output, fsx_mount, role_arn, image_uri, instance_type, script_preprocess, script_train):
    sagemaker_session = Session()
    # Preprocessing step
    processor = Processor(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        volume_size_in_gb=100,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session
    )
    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=s3_input, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=s3_output)],
        code=script_preprocess
    )
    # Training step
    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        volume_size=100,
        max_run=7200,
        input_mode='File',
        output_path=s3_output,
        sagemaker_session=sagemaker_session,
        subnets=[fsx_mount['subnet_id']],
        security_group_ids=[fsx_mount['security_group_id']],
        enable_network_isolation=True,
        encrypt_inter_container_traffic=True
    )
    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            'train': TrainingInput(s3_data=s3_output, input_mode='File')
        },
        code=script_train
    )
    pipeline = Pipeline(
        name="MedGPTSecurePipeline",
        steps=[preprocess_step, train_step],
        sagemaker_session=sagemaker_session,
        role=role_arn
    )
    pipeline.upsert(role_arn=role_arn)
    print("SageMaker pipeline created and ready.")
    return pipeline

if __name__ == "__main__":
    # Example usage (edit names as needed)
    bucket = "medgpt-secure-bucket-12345"
    role = "MedGPTSecureRole"
    trail = "MedGPTTrail"
    vpc_id = "vpc-xxxxxxxx"  # Replace with your VPC ID
    subnet_id = "subnet-xxxxxxxx"  # Replace with your subnet ID
    security_group_id = "sg-xxxxxxxx"  # Replace with your security group ID
    s3_import_path = f"s3://{bucket}/data"

    create_encrypted_bucket(bucket)
    create_iam_role(role)
    enable_cloudtrail(trail, bucket)
    # create_security_group('MedGPTSG', vpc_id)
    fsx_id = create_fsx_lustre("MedGPTFSx", s3_import_path, vpc_id, subnet_id, security_group_id)

    # SageMaker pipeline parameters (edit as needed)
    role_arn = "arn:aws:iam::123456789012:role/MedGPTSecureRole"  # Replace with your IAM role ARN
    image_uri = "<your-ecr-image-uri>"  # Replace with your container image URI
    instance_type = "ml.m5.2xlarge"
    script_preprocess = "preprocess.py"  # Your preprocessing script
    script_train = "train.py"  # Your training script
    fsx_mount = {'subnet_id': subnet_id, 'security_group_id': security_group_id}
    pipeline = create_sagemaker_pipeline(
        s3_input=s3_import_path,
        s3_output=f"s3://{bucket}/output",
        fsx_mount=fsx_mount,
        role_arn=role_arn,
        image_uri=image_uri,
        instance_type=instance_type,
        script_preprocess=script_preprocess,
        script_train=script_train
    )
    print("\nEnd-to-end secure SageMaker pipeline setup complete. Review all resources for compliance before production use.")
