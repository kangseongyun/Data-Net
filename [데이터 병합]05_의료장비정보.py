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
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')

file1 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\7.의료기관별상세정보서비스_05_의료장비정보_202309.csv", encoding='EUC-KR', dtype=str)
file2 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]에너지사용량_2018.xlsx", dtype=str)
# print(file1)
file1['mgm_bld_pk'] = file1['mgm_bld_pk'].str.split(',')
file1 = file1.explode('mgm_bld_pk')
file1 = file1.pivot_table(index=('암호화요양기호', '요양기관명', 'mgm_bld_pk'), columns=('장비코드명'), values='장비대수').reset_index()##'장비코드'제외
file1['총 장비대수'] = file1.iloc[:, 3:15].sum(axis=1)

merged_df = pd.merge(file1, file2, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='inner')
# print(merged_df)
merged_df.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보(에너지).csv", encoding='EUC-KR', index=False)

file3 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\1.병원정보서비스 2022.10..csv", encoding='EUC-KR', dtype=str)
# print(file3)
file3 = file3[~file3['종별코드명'].isin(['의원', '치과의원', '한의원','조산원'])]

file4 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보(에너지).csv", encoding='EUC-KR', dtype=str)
merged_df = pd.merge(file3, file4, on=['암호화요양기호', '요양기관명'])
# file3 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]에너지사용량_2018.xlsx", dtype=str)
# merged_df['mgm_bld_pk'] = merged_df['mgm_bld_pk'].str.split(',')
# merged_df1 = merged_df.explode('mgm_bld_pk')
# merged_df2 = pd.merge(merged_df1, file3, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='inner')
# # merged_df2 = pd.merge(merged_df1, file3, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='outer', indicator=True)
# # print(merged_df2)
# # tmerged_data2018 = merged_df2[merged_df2['_merge'] == 'both']
# # rnot_in_intersection_2018 = merged_df2[merged_df2['_merge'] == 'right_only']
# # lnot_in_intersection_2018 = merged_df2[merged_df2['_merge'] == 'left_only']
# # print(tmerged_data2018)
# # print(rnot_in_intersection_2018)
# # print(lnot_in_intersection_2018)
columns_to_remove = ['mgm_bld_pk']
merged_df.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보1(에너지).csv", encoding='EUC-KR', index=False)

merged_df2=merged_df.drop(columns=columns_to_remove)
merged_df2['사용량'] = merged_df2['사용량'].astype(float)
# rnot_in_intersection_2018['사용량'] = rnot_in_intersection_2018['사용량'].astype(float)
# lnot_in_intersection_2018['사용량'] = lnot_in_intersection_2018['사용량'].astype(float)
print(merged_df2)
merged_df2.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보1.csv", encoding='EUC-KR', index=False)



file5 = pd.read_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]건축물대장.xlsx', dtype=str)
file6 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보1.csv",  encoding='EUC-KR')

# 두 데이터프레임을 "매칭총괄표제부PK" 열을 기준으로 병합합니다.
merged_data2018 = pd.merge(file5, file6, left_on="매칭표제부PK", right_on="매칭총괄표제부PK", how="outer", indicator=True).drop(columns='매칭총괄표제부PK')
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
duplicates.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]B20181.xlsx', index=False)
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
# print(tmerged_data2018[['사용량', '용적률산정연면적(㎡)']].dtypes)


tmerged_data2018['용적률산정연면적(㎡)'] = pd.to_numeric(tmerged_data2018['용적률산정연면적(㎡)'], errors='coerce')
# print(tmerged_data2018[['사용량', '용적률산정연면적(㎡)']].dtypes)

tmerged_data2018 = tmerged_data2018.dropna(subset=['용적률산정연면적(㎡)'])  # Remove rows with NaN in '연면적(㎡)'
tmerged_data2018 = tmerged_data2018[tmerged_data2018['용적률산정연면적(㎡)'] != 0]
print(tmerged_data2018)
tmerged_data2018['사용량'] = pd.to_numeric(tmerged_data2018['사용량'], errors='coerce')
tmerged_data2018 = tmerged_data2018.dropna(subset=['사용량'])  # Remove rows with NaN in '연면적(㎡)'
tmerged_data2018 = tmerged_data2018[tmerged_data2018['사용량'] != 0]
tmerged_data2018['총의사수'] = pd.to_numeric(tmerged_data2018['총의사수'], errors='coerce')
tmerged_data2018 = tmerged_data2018.dropna(subset=['총의사수'])  # Remove rows with NaN in '연면적(㎡)'
tmerged_data2018 = tmerged_data2018[tmerged_data2018['총의사수'] != 0]
# print(tmerged_data2018)

tmerged_data2018['계산1'] = tmerged_data2018['사용량']/tmerged_data2018['용적률산정연면적(㎡)']
# tmerged_data2018['계산2'] = tmerged_data2018['총 장비대수']/tmerged_data2018['용적률산정연면적(㎡)']
tmerged_data2018['계산2'] = tmerged_data2018['총의사수']/tmerged_data2018['용적률산정연면적(㎡)']

print(tmerged_data2018)

tmerged_data2018.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 병합]05_의료장비정보.xlsx', index=False)


# result = merged_df2.groupby(list(merged_df2.columns[0:57]), dropna=False)['사용량'].sum().reset_index()
# # result1 = rnot_in_intersection_2018.groupby(list(rnot_in_intersection_2018.columns[0:57]), dropna=False)['사용량'].sum().reset_index()
# # result2 = lnot_in_intersection_2018.groupby(list(lnot_in_intersection_2018.columns[0:57]), dropna=False)['사용량'].sum().reset_index()
# # print(result)
# # print(result1)
# # print(result2)
#
# result.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]1.시설정보.csv", encoding='EUC-KR', index=False)
#
elec = tmerged_data2018[tmerged_data2018['에너지종류'].str.contains('전기', case=False, na=True)]
gas = tmerged_data2018[tmerged_data2018['에너지종류'].str.contains('도시가스', case=False, na=True)]
heat = tmerged_data2018[tmerged_data2018['에너지종류'].str.contains('지역난방', case=False, na=True)]
print(elec)
print(gas)
print(heat)

def graph(data, energy_type, unit):
    x1 = pd.to_numeric(data['계산2'], errors='coerce')
    y = pd.to_numeric(data['계산1'], errors='coerce')

    x = sm.add_constant(x1)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    # 산점도와 회귀선 그래프 생성
    plt.figure(figsize=(12, 10))
    plt.tick_params(axis='both', labelsize=20)
    plt.scatter(x['계산2'], y, label='Data')
    plt.plot(x['계산2'], model.predict(x), color='red', label='회귀선')
    # plt.xlabel('총장비대수/용적률산정연면적(대/㎡)', fontsize=20)
    plt.xlabel('총의사수/용적률산정연면적(명/㎡)', fontsize=20)

    plt.ylabel(f'사용량/용적률산정연면적 ({unit}/㎡)', fontsize=20)  # 수정된 부분
    plt.title(f'{energy_type}에 따른 그래프', fontsize=40)
    plt.tight_layout()

    data_count = len(data)
    plt.legend([f'Data (n={data_count})', '회귀선'], fontsize=16)

    # 그래프 저장
    plt.savefig(f'C:/Users/user/OneDrive - Ajou University/학부연구생 발표/데이터넷 과제/의료장비정보_의사{energy_type}.tiff', format='tiff', dpi=300)
    plt.show()

# 전기 그래프 그리기
graph(elec, '전기', 'kWh')

# 도시가스 그래프 그리기
graph(gas, '도시가스', 'MJ')

# 지역난방 그래프 그리기
graph(heat, '지역난방', 'Mcal')
