import torch.optim as optim
import torch
import torch.nn.functional as F
from normal_link_model import NormalLinkAE

class NormalLinkEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = NormalLinkAE().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print("✅ Normal-Link 모델 가중치 로드 완료")
        
        self.model.eval()

    def calculate_anomaly_score(self, image_tensor):
        """재구축 오차(MSE)를 통해 이상 점수 계산"""
        with torch.no_grad():
            reconstructed = self.model(image_tensor.to(self.device))
            
            # 원본과 복원 이미지의 차이 계산 (Pixel-wise MSE)
            loss = F.mse_loss(reconstructed, image_tensor.to(self.device), reduction='none')
            # 채널별 평균을 내어 히트맵(Error Map) 생성
            error_map = torch.mean(loss, dim=1).cpu().numpy()[0]
            # 전체 오차의 평균을 최종 이상 점수로 사용
            anomaly_score = torch.mean(loss).item()
            
        return anomaly_score, error_map

    def is_normal(self, image_tensor, threshold=0.02):
        """정상 여부 판단 (Triage 단계)"""
        score, _ = self.calculate_anomaly_score(image_tensor)
        print(f"📊 Anomaly Score: {score:.4f} (Threshold: {threshold})")
        
        # 점수가 임계값보다 낮으면 '정상'으로 판단
        return score < threshold, score