# Terraform template for secure MedGPT deployment (AWS example)
# HIPAA/GDPR/SOC2 best practices: encrypted storage, logging, IAM, network security

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "medgpt_secure" {
  bucket = "medgpt-secure-bucket-12345"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  versioning {
    enabled = true
  }
}

resource "aws_iam_role" "medgpt_role" {
  name = "MedGPTSecureRole"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_cloudtrail" "medgpt_trail" {
  name                          = "MedGPTTrail"
  s3_bucket_name                = aws_s3_bucket.medgpt_secure.id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true
}

# Add more resources as needed (EC2, security groups, etc.)
# Review and customize for your compliance requirements.
