import os
import ast
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

def prepare_mimic_df(aug_csv_path, chexpert_csv_path, img_root):
    labels_df = pd.read_csv(chexpert_csv_path)
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    aug_df = pd.read_csv(aug_csv_path)
    
    flat_data = []
    print(f"🔍 MIMIC 전체 데이터 파싱 중: {aug_csv_path}")
    missing_count = 0 

    for _, row in aug_df.iterrows():
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            
            # 🚀 [수정 포인트 1] 폴더 제한(p10~p13)을 풀고, 결측치('nan', '[]')만 깔끔하게 걸러냅니다.
            if raw_string == 'nan' or raw_string == '[]':
                continue
                
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    # 🚀 [수정 포인트 2] img_path 내부의 폴더명 검사 로직 삭제 완료
                    
                    img_full_path = os.path.join(img_root, img_path)
                    
                    if not os.path.exists(img_full_path):
                        missing_count += 1
                        continue

                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    
                    if not label_row.empty:
                        flat_data.append({
                            'path': img_full_path,
                            'labels': label_row[LABEL_ORDER].values[0]
                        })
            except Exception:
                continue
                
    final_df = pd.DataFrame(flat_data)
    print(f"✅ MIMIC 전체 파싱 완료: 총 {len(final_df)}장 (누락 {missing_count}장)")
    return final_df

class MimicDataset(Dataset):
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