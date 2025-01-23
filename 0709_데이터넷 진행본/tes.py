import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

merge_m18=pd.read_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토m2018.xlsx')







## 건축물대장과 에너지사용량 동일 경로 입력
base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

### 에너지사용량 excel 입력
filenames_energy = ["데이터넷_의료시설(에너지사용량_2018년).xlsx"]
sheetnames_energy = ['표제부_에너지사용량_계량기_2018년']

df_en = pd.DataFrame()
for i in range(0, len(filenames_energy)):
    file_path = os.path.join(base_dir, filenames_energy[i])
    df = pd.read_excel(file_path, sheet_name=sheetnames_energy[i]) # 엑셀 데이터 호출
    df_en = pd.concat([df_en, df], ignore_index=True)         # 엑셀 데이터 병합

df_en.rename(columns={'매칭총괄표제부PK': '매칭표제부PK'}, inplace=True) ## 기존 매칭총괄표제부PK를 매칭표제부PK로 정정
df_en['사용년월'] = pd.to_datetime(df_en['사용년월'], format='%Y%m')
df_en['year_use'] = df_en['사용년월'].dt.year           # 연도 구분
df_en['month_use'] = df_en['사용년월'].dt.month         # 월별 구분은 추후에 진행할 예정



print(df_en)