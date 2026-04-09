"""
SageMaker Training Job 실행 — Binary 2-class (Pleural Effusion vs Pneumothorax)
실행: python sagemaker/run_sagemaker-2class.py
"""
import boto3, os, tarfile, time
from pathlib import Path

# .env 파일 로드 (로컬 실행 시)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

AWS_REGION     = "ap-northeast-2"
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

BUCKET   = "say2-2team-bucket"
ROLE     = "arn:aws:iam::666803869796:role/SKKU_SageMaker_Role"
JOB_NAME = f"chexnet-binary2class-v{int(time.time())}"
IMAGE_URI = "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-v1.9"

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
s3 = session.client('s3')
sm = session.client('sagemaker')

# ── train-2class.py 패키징 & 업로드 ──
print("train-2class.py 패키징 중...")
script_dir = os.path.dirname(os.path.abspath(__file__))
tar_path = "/tmp/sourcedir-2class.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(os.path.join(script_dir, "train-2class.py"), arcname="train-2class.py")
s3.upload_file(tar_path, BUCKET, "code/sourcedir-2class.tar.gz")
print(f"업로드 완료: s3://{BUCKET}/code/sourcedir-2class.tar.gz")

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
                "S3Uri": f"s3://{BUCKET}/csv/3label_mimic_try1/",
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
        "S3OutputPath": f"s3://{BUCKET}/models/single2/",
    },
    ResourceConfig={
        "InstanceType": "ml.g4dn.16xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 200,
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 36000,   # 실제 학습 10시간
        "MaxWaitTimeInSeconds": 54000,  # 스팟 대기 포함 15시간
    },
    EnableManagedSpotTraining=True,
    CheckpointConfig={
        "S3Uri": f"s3://{BUCKET}/checkpoints/single2/",
        "LocalPath": "/opt/ml/checkpoints",
    },
    HyperParameters={
        "epochs":               "15",
        "batch-size":           "16",
        "lr":                   "1e-4",
        "early-stop-patience":  "5",
        "train-csv":            "/opt/ml/input/data/mimic_csv/single_2label_train.csv",
        "valid-csv":            "/opt/ml/input/data/mimic_csv/single_2label_valid.csv",

        "region":               AWS_REGION,
        "sagemaker_program":    "train-2class.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/code/sourcedir-2class.tar.gz",
    },
    Environment={
        "AWS_DEFAULT_REGION":    AWS_REGION,
        "AWS_ACCESS_KEY_ID":     AWS_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_KEY,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    Tags=[
        {"Key": "Project", "Value": "pre-sagemaker-2-2-team"},
    ]
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
    model_uri = f"s3://{BUCKET}/models/binary-2class/{JOB_NAME}/output/model.tar.gz"
    print(f"\n모델 위치: {model_uri}")
    print(f"\n다운로드 명령어:")
    print(f"  aws s3 cp {model_uri} mini_project/models/binary_2class.tar.gz --region {AWS_REGION}")
    print(f"  tar -xzf mini_project/models/binary_2class.tar.gz -C mini_project/models/")
else:
    print(f"실패 이유: {status.get('FailureReason', 'Unknown')}")
