import pandas as pd

labs_df = pd.read_csv('./data/labevents.csv',nrows=100)
print("📊 labevents 컬럼 목록:", labs_df.columns.tolist())