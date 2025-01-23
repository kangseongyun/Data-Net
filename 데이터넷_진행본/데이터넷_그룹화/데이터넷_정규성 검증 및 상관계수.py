import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.stats import kstest, shapiro, spearmanr
import seaborn as sns
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')


# 데이터 경로 및 파일명 설정
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)

# 엑셀 파일 읽기
q1 = pd.read_excel(file_path)

# 연도 설정
print('# of data1 : ',len(q1))

# 데이터 필터링
Group_Total = q1
Group_1 = q1[q1['종별코드명'].isin(['종합병원'])]
Group_2 = q1[q1['종별코드명'].isin(['병원','치과병원'])]
Group_3 = q1[q1['종별코드명'].isin(["요양병원"])]
Group_4 = q1[q1['종별코드명'].isin(["한방병원"])]
Group_5= q1[q1['종별코드명'].isin(['정신병원'])]

dir_path_porder1 = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본\그룹화"

year = 2022

# Group = Group_Total
# Group = Group_1
# Group = Group_2
# Group = Group_3
# Group = Group_4
Group = Group_5

Analysis_target = Group[Group['year_use'].isin([year])]







df_up = pd.DataFrame()
df_up['연면적'] = Analysis_target['연면적(㎡)']
df_up['층수'] = Analysis_target['지상층수']+Analysis_target['지하층수']
df_up['승강기수'] = Analysis_target['비상용승강기수'] + Analysis_target['승용승강기수']
df_up['의사수'] = Analysis_target['총의사수']
df_up['병상수'] = Analysis_target['총병상수']
df_up['USE_QTY_kWh'] = Analysis_target['USE_QTY_kWh']

if Group is Group_Total:
    filename1= 'total병원.xlsx'
elif Group is Group_1:
    filename1= '종합.xlsx'
elif Group is Group_2:
    filename1= '병원치과.xlsx'
elif Group is Group_3:
    filename1= '요양.xlsx'
elif Group is Group_4:
    filename1= '한방.xlsx'
elif Group is Group_5:
    filename1= '정신.xlsx'


file_path1 = os.path.join(dir_path_porder1, filename1)
df_up.to_excel(file_path1)








df_new = pd.DataFrame()
df_new['연면적'] = Analysis_target['연면적(㎡)']
df_new['용적률산정연면적'] = Analysis_target['용적률산정연면적(㎡)']
df_new['대지면적'] = Analysis_target['대지면적(㎡)']
df_new['건축면적'] = Analysis_target['건축면적(㎡)']
df_new['건폐율'] = Analysis_target['건폐율(%)']
df_new['용적률'] = Analysis_target['용적률(%)']
df_new['층수'] = Analysis_target['지상층수']+Analysis_target['지하층수']
df_new['지하층수'] = Analysis_target['지하층수']
df_new['지상층수'] = Analysis_target['지상층수']
df_new['승강기수'] = Analysis_target['비상용승강기수'] + Analysis_target['승용승강기수']
df_new['의사수'] = Analysis_target['총의사수']
df_new['병상수'] = Analysis_target['총병상수']

df_new['USE_QTY_kWh'] = Analysis_target['USE_QTY_kWh']

df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
print('# of data1 : ',len(df_new))

df_new = df_new[(df_new['대지면적']>0)&(df_new['의사수']>0)&(df_new['지상층수']>0)]
print('# of data1 : ',len(df_new))


df = pd.DataFrame(df_new)


# KS test와 SW test 결과 저장을 위한 리스트 생성
ks_test_results = []
sw_test_results = []

# KS test와 SW test 수행 및 결과 저장
for column in df.columns:
    ks_stat, ks_pvalue = kstest(df[column], 'norm', args=(df[column].mean(), df[column].std()))
    sw_stat, sw_pvalue = shapiro(df[column])
    ks_test_results.append({'Column': column, 'KS Statistic': ks_stat, 'KS p-value': ks_pvalue})
    sw_test_results.append({'Column': column, 'SW Statistic': sw_stat, 'SW p-value': sw_pvalue})

# 결과를 데이터프레임으로 변환
ks_test_df = pd.DataFrame(ks_test_results)
sw_test_df = pd.DataFrame(sw_test_results)



print("KS Test Results:")
print(ks_test_df)

print("\nSW Test Results:")
print(sw_test_df)




pearson_corr = df.corr(method='pearson')
if Group is Group_Total:
    filename = 'group_5_spearman_corr.xlsx'
elif Group is Group_1:
    filename = 'group_1_pearson_corr.xlsx'
elif Group is Group_2:
    filename = 'group_2_pearson_corr.xlsx'
elif Group is Group_3:
    filename = 'group_3_pearson_corr.xlsx'
elif Group is Group_4:
    filename = 'group_4_pearson_corr.xlsx'
elif Group is Group_5:
    filename = 'group_5_pearson_corr.xlsx'

file_path = os.path.join(dir_path_porder1, filename)
pearson_corr.to_excel(file_path)


spearman_corr = df.corr(method='spearman')
if Group is Group_Total:
    filename = 'total_group_spearman_corr.xlsx'
elif Group is Group_1:
    filename = 'group_1_spearman_corr.xlsx'
elif Group is Group_2:
    filename = 'group_2_spearman_corr.xlsx'
elif Group is Group_3:
    filename = 'group_3_spearman_corr.xlsx'
elif Group is Group_4:
    filename = 'group_4_spearman_corr.xlsx'
elif Group is Group_5:
    filename = 'group_5_spearman_corr.xlsx'


file_path = os.path.join(dir_path_porder1, filename)
spearman_corr.to_excel(file_path)

# data=pd.DataFrame()
# data['skew']=df.skew()
# data['kurtosis'] = df.kurtosis()
# data=data.reset_index()
# print(data)
# data=pd.concat([data,ks_test_df,sw_test_df],axis=1)
# filename = 'test.xlsx'
# file_path = os.path.join(dir_path_porder1, filename)
# data.to_excel(file_path)
# print(data)


# corr=pearson_corr

corr=spearman_corr



mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask=mask, cmap='Reds', square=True, linewidths=.5, vmin=-1, vmax=1, fmt=".2f", annot_kws={"color": "white", "fontsize": 8})

plt.tight_layout()
plt.show()
