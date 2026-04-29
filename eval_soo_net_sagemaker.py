import boto3, os, tarfile, time
from pathlib import Path

# ── 설정 ──
AWS_REGION = "ap-northeast-2"
ROLE = "arn:aws:iam::666803869796:role/SKKU_SageMaker_Role"
BUCKET = "say2-2team-bucket"
JOB_NAME = f"soonet-uones-eval-only-{int(time.time())}"
IMAGE_URI = "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"

# U-Ones 모델 가중치가 저장된 S3 경로 (학습 결과물 output)
# soonet_uones.pth 는 모델 출력 tar.gz 안에 있음
UONES_MODEL_S3 = f"s3://{BUCKET}/models/soonet_final_results/soonet-pure-ones-final-exec-1776238400/output/model.tar.gz"

# ── 1. 코드 패키징 및 S3 업로드 ──
print("📦 평가 코드를 패키징하여 S3에 업로드 중...")
s3_client = boto3.client('s3', region_name=AWS_REGION)
source_path = "."
tar_path = "/tmp/soonet_eval_exec.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(source_path, arcname=".")
s3_client.upload_file(tar_path, BUCKET, "code/soonet_eval_exec.tar.gz")
print(f"✅ 코드 업로드 완료: s3://{BUCKET}/code/soonet_eval_exec.tar.gz")

# ── 2. SageMaker 평가 작업 생성 ──
print(f"🚀 U-Ones 평가 전용 작업 시작: {JOB_NAME}")
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
        },
        {
            # U-Ones 모델 가중치 (soonet_uones.pth) 가 들어있는 tar.gz
            "ChannelName": "pretrained",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/models/soonet_final_results/soonet-pure-ones-final-exec-1776238400/output/",
            }},
        },
    ],
    OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/models/soonet_eval_results/"},
    ResourceConfig={
        "InstanceType": "ml.g4dn.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    },
    StoppingCondition={"MaxRuntimeInSeconds": 14400},  # 4시간 (평가만이므로 충분)
    EnableManagedSpotTraining=False,
    HyperParameters={
        "eval-csv-name": "chexpert_balanced_u_ones.csv",
        "model-filename": "soonet_uones.pth",
        "unet-weight-path": "unet_lung_mask_ep10.pth",
        "sagemaker_program": "eval_soo_net_cloud.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/code/soonet_eval_exec.tar.gz",
    },
    Tags=[{"Key": "project", "Value": "pre-sagemaker-2-2-team"}]
)

print(f"✅ 평가 작업 제출 완료: {JOB_NAME}")
print(f"🔗 확인 링크: https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs/{JOB_NAME}")
print(f"📁 결과 저장 경로: s3://{BUCKET}/models/soonet_eval_results/{JOB_NAME}/output/")
