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

file1 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\1.병원정보서비스 2022.10..csv", encoding='EUC-KR', dtype=str)
file2 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\4.의료기관별상세정보서비스_02_세부정보_202309.csv", encoding='EUC-KR', dtype=str)
print(file2)

merged_df = pd.merge(file1, file2, on=['암호화요양기호', '요양기관명'])

file3 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]에너지사용량_2018.xlsx", dtype=str)
merged_df['mgm_bld_pk'] = merged_df['mgm_bld_pk'].str.split(',')
merged_df1 = merged_df.explode('mgm_bld_pk')
merged_df2 = pd.merge(merged_df1, file3, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='inner')
# merged_df2 = pd.merge(merged_df1, file3, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='outer', indicator=True)
# print(merged_df2)
# tmerged_data2018 = merged_df2[merged_df2['_merge'] == 'both']
# rnot_in_intersection_2018 = merged_df2[merged_df2['_merge'] == 'right_only']
# lnot_in_intersection_2018 = merged_df2[merged_df2['_merge'] == 'left_only']
# print(tmerged_data2018)
# print(rnot_in_intersection_2018)
# print(lnot_in_intersection_2018)
columns_to_remove = ['mgm_bld_pk', 'mgm_upper_bld_pk']
merged_df2=merged_df2.drop(columns=columns_to_remove)
merged_df2['사용량'] = merged_df2['사용량'].astype(float)
merged_df2.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]2.세부정보1.csv", encoding='EUC-KR', index=False)

result = merged_df2.groupby(list(merged_df2.columns[0:62]), dropna=False)['사용량'].sum().reset_index()
result.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]2.세부정보.csv", encoding='EUC-KR', index=False)

elec = result[result['에너지종류'].str.contains('전기', case=False, na=True)]
gas = result[result['에너지종류'].str.contains('도시가스', case=False, na=True)]
heat = result[result['에너지종류'].str.contains('지역난방', case=False, na=True)]

def graph(A,B):
    # result = elec[elec['총의사수'] != 0]
    x1=pd.to_numeric(A['총의사수'], errors='coerce')
    y = pd.to_numeric(A['사용량'], errors='coerce')


    x = sm.add_constant(x1)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    # 산점도와 회귀선 그래프 생성
    plt.figure(figsize=(12, 10))  # 가로 8인치, 세로 6인치 크기로 조절
    plt.tick_params(axis='both', labelsize=20)  # xtick 크기 조절
    plt.scatter(x['총의사수'], y, label='Data')
    plt.plot(x['총의사수'], model.predict(x), color='red', label='회귀선')
    plt.xlabel('총의사수', fontsize=20)
    plt.ylabel('사용량'+B, fontsize=20)
    plt.tight_layout()
    plt.legend()
    return plt.show()

graph(elec,'(전기,kWh)')
graph(gas,'(도시가스,MJ)')
graph(heat,'(지역난방,Mcal)')