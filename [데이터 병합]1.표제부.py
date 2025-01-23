import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
import openpyxl
import xlsxwriter


file1 = pd.read_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]건축물대장.xlsx', dtype=str)
file2 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보1.csv",  encoding='EUC-KR')

# 두 데이터프레임을 "매칭총괄표제부PK" 열을 기준으로 병합합니다.
merged_data2018 = pd.merge(file1, file2, left_on="매칭표제부PK", right_on="매칭총괄표제부PK", how="outer", indicator=True).drop(columns='매칭총괄표제부PK')
print(merged_data2018)
# 각 파일을 file1과 병합하고 indicator 열을 생성합니다.
tmerged_data2018 = merged_data2018[merged_data2018['_merge'] == 'both']
print(tmerged_data2018)

# rnot_in_intersection_2018 = merged_data2018[merged_data2018['_merge'] == 'right_only']
# # print(rnot_in_intersection_2018)
#
# lnot_in_intersection_2018 = merged_data2018[merged_data2018['_merge'] == 'left_only']
# # print(lnot_in_intersection_2018)

tmerged_data2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]B2018.xlsx', index=False)
duplicates = tmerged_data2018[tmerged_data2018.duplicated(subset=['암호화요양기호', '요양기관명','사용년월', '에너지종류', '단위명'], keep=False)]
# duplicates.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]B20181.xlsx', index=False)
duplicates_titles = duplicates['매칭표제부PK']

# 중복된 행의 인덱스를 가져옵니다.
duplicates_indices = duplicates.index

# 중복된 행을 제거합니다.
tmerged_data2018 = tmerged_data2018.drop(index=duplicates_indices)

print(tmerged_data2018)

# rnot_in_intersection_2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]r2018.xlsx', index=False)
#
# lnot_in_intersection_2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]l2018.xlsx', index=False)
# result = merged_df2.groupby(list(merged_df2.columns[1:57]), dropna=False)['사용량'].sum().reset_index()
print(tmerged_data2018[['사용량', '용적률산정연면적(㎡)']].dtypes)


tmerged_data2018['용적률산정연면적(㎡)'] = pd.to_numeric(tmerged_data2018['용적률산정연면적(㎡)'], errors='coerce')
print(tmerged_data2018[['사용량', '용적률산정연면적(㎡)']].dtypes)

tmerged_data2018 = tmerged_data2018.dropna(subset=['용적률산정연면적(㎡)'])  # Remove rows with NaN in '연면적(㎡)'
tmerged_data2018 = tmerged_data2018[tmerged_data2018['용적률산정연면적(㎡)'] != 0]
print(tmerged_data2018)
tmerged_data2018['사용량'] = pd.to_numeric(tmerged_data2018['사용량'], errors='coerce')
tmerged_data2018 = tmerged_data2018.dropna(subset=['사용량'])  # Remove rows with NaN in '연면적(㎡)'
# tmerged_data2018 = tmerged_data2018[tmerged_data2018['사용량'] != 0]
# print(tmerged_data2018)

tmerged_data2018['계산'] = tmerged_data2018['사용량']/tmerged_data2018['용적률산정연면적(㎡)']
print(tmerged_data2018)

tmerged_data2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]a2018.xlsx', index=False)

