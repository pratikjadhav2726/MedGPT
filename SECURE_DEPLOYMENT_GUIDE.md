# Secure Deployment Guide for MedGPT (HIPAA, GDPR, SOC2)

This guide provides step-by-step instructions for deploying MedGPT in a compliant manner using both Python SDK (boto3) and Terraform.

## Prerequisites
- AWS account (or adapt for GCP/Azure)
- AWS CLI configured (`aws configure`)
- Python 3.x and `boto3` installed
- Terraform installed

## 1. Python SDK Deployment

1. Edit `secure_deploy.py` to set unique bucket names, role names, and your VPC ID.
2. Run:
   ```bash
   pip install boto3
   python secure_deploy.py
   ```
3. Review created resources in AWS Console for compliance.

## 2. Terraform Deployment

1. Edit `secure_deploy.tf` to set unique resource names and region.
2. Initialize and apply:
   ```bash
   terraform init
   terraform apply
   ```
3. Review resources in AWS Console.

## 3. Compliance Checklist
- All data at rest is encrypted (S3, EBS, RDS, etc.)
- IAM roles follow least privilege
- CloudTrail logging is enabled
- Security groups restrict access
- Audit logs are retained and monitored
- Data de-identification is implemented where required
- Breach notification and incident response plans are in place

## 4. References
- See `HIPAA_GDPR_Compliance_MedGPT.pdf` for detailed requirements
- Review AWS/GCP/Azure compliance documentation

## 5. Notes
- These scripts/templates are starting points. Review and adapt for your specific regulatory and organizational requirements.
- Consult with your compliance and security teams before production deployment.
