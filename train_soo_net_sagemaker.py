import boto3, os, tarfile, time # AWS SDK 및 파일 처리를 위한 라이브러리 임포트
from pathlib import Path # 경로 처리를 위한 라이브러리 임포트

# ── 1. 실험 대조군 설정 (온디맨드 강제 실행 모드) ──
TRAIN_CSV = "chexpert_balanced_manual1.csv"
LABEL_POLICY = "mixed"

AWS_REGION = "ap-northeast-2" # AWS 서비스 리전 설정 (서울)
ROLE = "arn:aws:iam::666803869796:role/SKKU_SageMaker_Role" # SageMaker 실행 권한 역할 ARN
BUCKET = "say2-2team-bucket" # S3 버킷 이름
JOB_NAME = f"soonet-mixed-policy-exec-{int(time.time())}" 
IMAGE_URI = "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"

# ── 2. 학습 코드 패키징 및 S3 업로드 ──
print(f"📦 최종 검증된 코드를 업로드 중입니다...")
s3_client = boto3.client('s3', region_name=AWS_REGION)
source_path = "." 
tar_path = "/tmp/soonet_final_exec.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(source_path, arcname=".") 
s3_client.upload_file(tar_path, BUCKET, f"code/soonet_final_exec.tar.gz")

# ── 3. SageMaker 학습 작업(Training Job) 생성 ──
print(f"🚀 [온디맨드 강제 실행] 클라우드 학습 시작: {JOB_NAME}")
sm_client = boto3.client('sagemaker', region_name=AWS_REGION)
sm_client.create_training_job(
    TrainingJobName=JOB_NAME,
    AlgorithmSpecification={
        "TrainingImage": IMAGE_URI,
        "TrainingInputMode": "FastFile",
    },
    RoleArn=ROLE,
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/final_csv/", 
            }},
        },
        {
            "ChannelName": "images",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/cheXpert_data/dataset_resized_448/",
            }},
        },
        {
            "ChannelName": "mimic",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/data/mimic-cxr-448/",
            }},
        }
    ],
    OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/models/soonet_final_results/"},
    ResourceConfig={
        "InstanceType": "ml.g4dn.xlarge", 
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    },
    StoppingCondition={"MaxRuntimeInSeconds": 172800},
    # 💡 [긴급 변경] 리소스 한도 충돌을 피하기 위해 온디맨드(False)로 실행합니다.
    EnableManagedSpotTraining=False, 
    HyperParameters={
        "epochs": "10",
        "batch-size": "16",
        "learning-rate": "0.0001",
        "train-csv-name": TRAIN_CSV, 
        "label-policy": LABEL_POLICY,
        "sagemaker_program": "train_soo_net.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/code/soonet_final_exec.tar.gz",
    },
    Tags=[{"Key": "project", "Value": "pre-sagemaker-2-2-team"}]
)

print(f"✅ 요청 성공! ARN: {JOB_NAME}")
print(f"🔗 확인 링크: https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs/{JOB_NAME}")
