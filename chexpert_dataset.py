import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def prepare_chexpert_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    
    flat_data = []
    print(f"🔍 CheXpert 데이터 파싱 중: {csv_path}")
    missing_count = 0

    for _, row in df.iterrows():
        img_full_path = os.path.join(img_root, row['Path'])
        if not os.path.exists(img_full_path):
            missing_count += 1
            continue
            
        flat_data.append({
            'path': img_full_path,
            'labels': row[LABEL_ORDER].values
        })
        
    final_df = pd.DataFrame(flat_data)
    print(f"✅ CheXpert 파싱 완료: 총 {len(final_df)}장 (누락 {missing_count}장)")
    return final_df

class ChexpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = row['path']
        
        try:
            # 1채널 흑백(Grayscale) 모드로 이미지를 엽니다
            image = Image.open(img_full_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'].astype(float))
        
        if self.transform:
            image = self.transform(image)
        return image, label