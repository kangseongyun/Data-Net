import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os


plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')

file1 = pd.read_csv(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병원정보서비스_filtered.csv", encoding='utf-8-sig')
print(file1)
file1.to_csv(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\검토용.csv", encoding='UTF-8-SIG')
# file2 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(건축물대장).xlsx", sheet_name="층별개요", dtype=str)
# print(file2)
# file3 = file1[file1['mgm_bld_pk'].isin(file2['매칭표제부PK'].unique())]
# print(file3)
# file3.to_csv(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\검토용.csv", encoding='UTF-8-SIG')
# df_combined = pd.merge(left=file3, right=file2, how='inner', on=None, left_on='mgm_bld_pk', right_on='매칭표제부PK', left_index=False, right_index=False, suffixes=('_x', '_y'))
#
# print(df_combined)
# df_combined.to_csv(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\검토용.csv", encoding='UTF-8-SIG')
#



dir_path = r"C:\Users\user\Desktop\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결"
dir_path1 = r"C:\Users\user\Desktop\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\병원정보 편집본"

filename = "1.병원정보서비스 2022.10..csv"
file_path = os.path.join(dir_path,filename)
hos_00 = pd.read_csv(file_path)
file_path1 = os.path.join(dir_path1,filename)
hos_00.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "3.의료기관별상세정보서비스_01_시설정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_01 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_01.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "4.의료기관별상세정보서비스_02_세부정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_02 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_02.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_03 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_03.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_05 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_05.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_06 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_06.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_07 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_07.to_csv(file_path1, encoding='UTF-8-SIG')

filename = "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_08 = pd.read_csv(file_path, encoding='cp949')
file_path1 = os.path.join(dir_path1,filename)
hos_08.to_csv(file_path1, encoding='UTF-8-SIG')
