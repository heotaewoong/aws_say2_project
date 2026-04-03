"""
SageMaker Training Job 실행 — MIMIC-CXR 448x448 only
실행: python sagemaker/run_sagemaker.py
"""
import boto3, os, tarfile, time

AWS_REGION    = "ap-northeast-2"
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

BUCKET   = "say2-2team-bucket"
ROLE     = "arn:aws:iam::666803869796:role/SKKU_SageMaker_Role"
JOB_NAME = f"chexnet-mimic448-v{int(time.time())}"
IMAGE_URI = "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-v1.9"

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
s3 = session.client('s3')
sm = session.client('sagemaker')

# ── train.py 패키징 & 업로드 ──
print("train.py 패키징 중...")
script_dir = os.path.dirname(os.path.abspath(__file__))
tar_path = "/tmp/sourcedir.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(os.path.join(script_dir, "train.py"), arcname="train.py")
s3.upload_file(tar_path, BUCKET, "code/sourcedir.tar.gz")
print(f"업로드 완료: s3://{BUCKET}/code/sourcedir.tar.gz")

# ── Training Job 생성 ──
print(f"\nTraining Job 시작: {JOB_NAME}")
response = sm.create_training_job(
    TrainingJobName=JOB_NAME,
    AlgorithmSpecification={
        "TrainingImage": IMAGE_URI,
        "TrainingInputMode": "File",
    },
    RoleArn=ROLE,
    InputDataConfig=[
        {
            "ChannelName": "mimic_images",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/data/mimic-cxr-448/",
                "S3DataDistributionType": "FullyReplicated",
            }},
        },
        {
            "ChannelName": "mimic_csv",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/csv/",
                "S3DataDistributionType": "FullyReplicated",
            }},
        },
        {
            "ChannelName": "code",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/code/",
                "S3DataDistributionType": "FullyReplicated",
            }},
        },
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{BUCKET}/models/mimic-only/",
    },
    ResourceConfig={
        "InstanceType": "ml.g4dn.16xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 200,
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 43200,   # 6시간
        "MaxWaitTimeInSeconds": 50400,  # 스팟 대기 8시간
    },
    EnableManagedSpotTraining=True,
    CheckpointConfig={
        "S3Uri": f"s3://{BUCKET}/checkpoints/chexnet-mimic-v2/",
        "LocalPath": "/opt/ml/checkpoints",
    },
    HyperParameters={
        "epochs":               "15",
        "batch-size":           "16",
        "lr":                   "1e-4",
        "early-stop-patience":  "5",
        "region":               AWS_REGION,
        "mimic-bucket":         BUCKET,
        "mimic-max-samples":    "20000",
        "sagemaker_program":    "train.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/code/sourcedir.tar.gz",
    },
    Environment={
        "AWS_DEFAULT_REGION":    AWS_REGION,
        "AWS_ACCESS_KEY_ID":     AWS_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_KEY,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

print(f"Job ARN: {response['TrainingJobArn']}")
print(f"콘솔: https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/jobs/{JOB_NAME}")

# ── 상태 모니터링 ──
print("\n모니터링 중... (Ctrl+C로 중단해도 Job은 계속 실행됨)")
while True:
    status = sm.describe_training_job(TrainingJobName=JOB_NAME)
    state     = status['TrainingJobStatus']
    secondary = status.get('SecondaryStatus', '')
    print(f"  [{time.strftime('%H:%M:%S')}] {state} / {secondary}    ", end='\r')
    if state in ('Completed', 'Failed', 'Stopped'):
        print(f"\n최종 상태: {state}")
        break
    time.sleep(30)

if state == 'Completed':
    model_uri = f"s3://{BUCKET}/models/mimic-only/{JOB_NAME}/output/model.tar.gz"
    print(f"\n모델 위치: {model_uri}")
    print(f"\n다운로드 명령어:")
    print(f"  aws s3 cp {model_uri} mini_project/models/chexnet_mimic448.tar.gz --region {AWS_REGION}")
    print(f"  tar -xzf mini_project/models/chexnet_mimic448.tar.gz -C mini_project/models/")
else:
    print(f"실패 이유: {status.get('FailureReason', 'Unknown')}")
