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

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')

file1 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(건축물대장).xlsx", sheet_name="건축물대장(총괄표제부)", dtype=str)
file2 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2018년).xlsx", sheet_name="총괄표제부_에너지사용량_계량기_2018년", dtype=str)
file3 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2019년).xlsx", sheet_name="총괄표제부_에너지사용량_계량기_2019년", dtype=str)
file4 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2020년).xlsx", sheet_name="총괄표제부_에너지사용량_계량기_2020년", dtype=str)
file5 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2021년).xlsx", sheet_name="총괄표제부_에너지사용량_계량기_2021년", dtype=str)
file6 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2022년).xlsx", sheet_name="총괄표제부_에너지사용량_계량기_2022년", dtype=str)


# 두 데이터프레임을 "매칭총괄표제부PK" 열을 기준으로 병합합니다.
merged_data2018 = pd.merge(file1, file2, on="매칭총괄표제부PK", how="outer", indicator=True)
merged_data2019 = pd.merge(file1, file3, on="매칭총괄표제부PK", how="outer", indicator=True)
merged_data2020 = pd.merge(file1, file4, on="매칭총괄표제부PK", how="outer", indicator=True)
merged_data2021 = pd.merge(file1, file5, on="매칭총괄표제부PK", how="outer", indicator=True)
merged_data2022 = pd.merge(file1, file6, on="매칭총괄표제부PK", how="outer", indicator=True)

# 각 파일을 file1과 병합하고 indicator 열을 생성합니다.
tmerged_data2018 = pd.merge(file1, file2, on="매칭총괄표제부PK", how="inner", indicator=True)
tmerged_data2019 = pd.merge(file1, file3, on="매칭총괄표제부PK", how="inner", indicator=True)
tmerged_data2020 = pd.merge(file1, file4, on="매칭총괄표제부PK", how="inner", indicator=True)
tmerged_data2021 = pd.merge(file1, file5, on="매칭총괄표제부PK", how="inner", indicator=True)
tmerged_data2022 = pd.merge(file1, file6, on="매칭총괄표제부PK", how="inner", indicator=True)

rnot_in_intersection_2018 = merged_data2018[merged_data2018['_merge'] == 'right_only']
rnot_in_intersection_2019 = merged_data2019[merged_data2019['_merge'] == 'right_only']
rnot_in_intersection_2020 = merged_data2020[merged_data2020['_merge'] == 'right_only']
rnot_in_intersection_2021 = merged_data2021[merged_data2021['_merge'] == 'right_only']
rnot_in_intersection_2022 = merged_data2022[merged_data2022['_merge'] == 'right_only']

lnot_in_intersection_2018 = merged_data2018[merged_data2018['_merge'] == 'left_only']
lnot_in_intersection_2019 = merged_data2019[merged_data2019['_merge'] == 'left_only']
lnot_in_intersection_2020 = merged_data2020[merged_data2020['_merge'] == 'left_only']
lnot_in_intersection_2021 = merged_data2021[merged_data2021['_merge'] == 'left_only']
lnot_in_intersection_2022 = merged_data2022[merged_data2022['_merge'] == 'left_only']

rnot_in_intersection_2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\right\r2018.xlsx', index=False)
rnot_in_intersection_2019.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\right\r2019.xlsx', index=False)
rnot_in_intersection_2020.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\right\r2020.xlsx', index=False)
rnot_in_intersection_2021.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\right\r2021.xlsx', index=False)
rnot_in_intersection_2022.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\right\r2022.xlsx', index=False)

lnot_in_intersection_2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\left\l2018.xlsx', index=False)
lnot_in_intersection_2019.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\left\l2019.xlsx', index=False)
lnot_in_intersection_2020.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\left\l2020.xlsx', index=False)
lnot_in_intersection_2021.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\left\l2021.xlsx', index=False)
lnot_in_intersection_2022.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\left\l2022.xlsx', index=False)

def tap(A,B):
    file7 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2022년).xlsx", sheet_name="에너지용도코드", dtype=str)
    # file7에서 "사용용도코드"와 "에너지종류" 열을 선택합니다.
    mapping_data = file7[['용도코드', '에너지종류']]
    # "사용용도코드"를 "용도코드"와 비교하여 동일한 경우 "에너지종류"로 수정합니다.
    A['에너지종류'] = A.apply(lambda row: mapping_data[mapping_data['용도코드'] == row['사용용도코드']]['에너지종류'].values[0] if row['사용용도코드'] in mapping_data['용도코드'].values else '', axis=1)
    file8 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2022년).xlsx", sheet_name="단위코드", dtype=str)
    mapping_data = file8[['단위코드', '단위명']]
    A['단위명'] = A.apply(lambda row: mapping_data[mapping_data['단위코드'] == row['단위코드']]['단위명'].values[0] if row['단위코드'] in mapping_data['단위코드'].values else '', axis=1)
    C=A.to_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]총괄표제부\병합데이터\[데이터 병합]총괄표제부_"+str(B)+"_with_에너지종류.xlsx", index=False)
    return C

data2018=tap(tmerged_data2018,2018)
data2019=tap(tmerged_data2019,2019)
data2020=tap(tmerged_data2020,2020)
data2021=tap(tmerged_data2021,2021)
data2022=tap(tmerged_data2022,2022)