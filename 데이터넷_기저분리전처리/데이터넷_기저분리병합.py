import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_dir1=r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_텍스트파일"
file01='데이터넷_아주대1__1_서울경기_2018_2019년_총괄표제부_계량기별_사용량_용도전체.txt'
file02='데이터넷_아주대1__2_서울경기_2018_2019년_총괄표제부_에너지원별_사용량_건물용도.txt'
file03='데이터넷_아주대1__3_서울경기_2018_2019년_표제부_계량기별_사용량_용도전체.txt'
file04='데이터넷_아주대1__4_서울경기_2018_2019년_표제부_에너지원별_사용량_건물용도.txt'
file05='데이터넷_아주대1__5_서울경기_의료기관_총괄표제부.txt'
file06='데이터넷_아주대1__6_서울경기_의료기관_표제부.txt'
file07='데이터넷_아주대1__7_서울경기_의료기관_층별개요.txt'

file_path_1= os.path.join(base_dir1, file01)
file_path_2= os.path.join(base_dir1, file02)
file_path_3= os.path.join(base_dir1, file03)
file_path_4= os.path.join(base_dir1, file04)
file_path_5= os.path.join(base_dir1, file05)
file_path_6= os.path.join(base_dir1, file06)
file_path_7= os.path.join(base_dir1, file07)

df01= pd.read_table(file_path_1,sep='|', low_memory=False)
df02= pd.read_table(file_path_2,sep='|', low_memory=False)
df03= pd.read_table(file_path_3,sep='|', low_memory=False)
df04= pd.read_table(file_path_4,sep='|', low_memory=False)
df05= pd.read_table(file_path_5,sep='|', low_memory=False)
df06= pd.read_table(file_path_6,sep='|', low_memory=False)
df07= pd.read_table(file_path_7,sep='|', low_memory=False)



### 6_서울경기_의료기관_표제부 ############################################################################################

## 기본
print('6_서울경기_의료기관_표제부')
print("# of PK0 : ", df06['mgm_bld_pk'].nunique())
print('# of data0 : ', df06.shape[0])
print(' ')

## 총괄표제부 NaN만 추출
bld_pk1=df06
bld_pk1 = bld_pk1[bld_pk1['mgm_upper_bld_pk'].isna()]

print('mgm_upper_bld_pk의 NaN제거')
print("# of PK1 : ", bld_pk1['mgm_bld_pk'].nunique())
print('# of data1 : ', bld_pk1.shape[0])
print(' ')


## 일반건축물 추출
bld_pk2=bld_pk1
bld_pk2=bld_pk2[bld_pk2['regstr_gb_nm']=='일반']
print('regstr_gb_nm = 일반')
print("# of PK2 : ", bld_pk2['mgm_bld_pk'].nunique())
print('# of data2 : ', bld_pk2.shape[0])
print(' ')


bld_pk3=bld_pk2
bld_pk3 = bld_pk3[(bld_pk3['vl_rat_estm_totarea']>0)&(bld_pk3['totarea']>0)]
print('용적률산정연면적 & 연면적 > 0')
print("# of PK3 : ", bld_pk3['mgm_bld_pk'].nunique())
print('# of data3 : ', bld_pk3.shape[0])
print(' ')

bld_pk3.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\건축_표제부필터.xlsx')
bld_pk_a = bld_pk3[bld_pk3.duplicated(['mgm_bld_pk'], keep=False)]
bld_pk_a.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\중복.xlsx')





print(' ')
print("3_계량기별_사용량 ############################################################################################")

### 3_계량기별_사용량 ####################################################################################################

## 기본
energy_m=df03
print('3_계량기별_사용량')
print("# of PK : ", energy_m['mgm_bld_pk'].nunique())
print('# of data : ', energy_m.shape[0])
print(' ')



def year_separation(A):
    A['use_ym'] = pd.to_datetime(A['use_ym'], format='%Y%m')
    A['year_use'] = A['use_ym'].dt.year           # 연도 구분
    A['month_use'] = A['use_ym'].dt.month         # 월별 구분은 추후에 진행할 예정
    return A
energy_m=year_separation(energy_m)

# PK_list = bld_pk3['mgm_bld_pk'].unique()
# energy_m = energy_m[energy_m['mgm_bld_pk'].isin(PK_list)]
# print('건축물대장 필터')
# print("# of PK0 : ", energy_m['mgm_bld_pk'].nunique())
# print('# of data0 : ', energy_m.shape[0])
# print(' ')

## 건물용도 해당만 필터
energy_m1=energy_m
energy_m1=energy_m1[energy_m1['bld_purps_flag']=='Y']
print('bld_purps_flag ＝ Y')
print("# of PK1 : ", energy_m1['mgm_bld_pk'].nunique())
print('# of data1 : ', energy_m1.shape[0])
print(' ')



## 일반건축물 추출
energy_m2=energy_m1
energy_m2=energy_m2[energy_m2['regstr_gb_nm']=='일반']
print('regstr_gb_nm ＝ 일반')
print("# of PK2 : ", energy_m2['mgm_bld_pk'].nunique())
print('# of data2 : ', energy_m2.shape[0])
print(' ')



## 에너지원별 kWh기준 단위 통일(1: KWh 과 8: MJ 만 존재)
def energy_conversion(row):
    if row['unit_cd'] == 1:
        x = 1
    elif row['unit_cd'] == 2:
        x = 42.7 * 1 / 3.6
    elif row['unit_cd'] == 3:
        x = 1 / 0.860 * 1000
    elif row['unit_cd'] == 4:
        x = 1000
    elif row['unit_cd'] == 6:
        x = 1 / 0.860
    elif row['unit_cd'] == 8:
        x = 1 / 3.6
    else:
        x = 63.4 * 1 / 3.6  # UNIT_CD = 14 #
    row['eb_qty'] = x * row['eb_qty']            # 기저 소비량
    row['ec_qty'] = x * row['ec_qty']            # 냉방 소비량
    row['eh_qty'] = x * row['eh_qty']            # 난방 소비량
    row['tot_use_qty'] = x * row['tot_use_qty']  # 합계 소비량
    return row

# ### energy conversion
# - 1 Nm3 = 10.55 kWh
# - 1 gcal = 42.7*1/3.6 kWh
# - 1 MWh = 1000 kWh
# - 1 Mcal = 1/0.860 kWh
# - 1 MJ = 1/3.6 kWh
# - 1 Nm3 = 63.4*1/3.6

energy_m2 = energy_m2.apply(energy_conversion, axis=1)



## 12개월 모두 존재 데이터만 추출(11:전기, 12:가스, 13:지역냉난방)
energy_m3=energy_m2
def divide_energy(row):
    if row['engy_kind_cd'] == 11:
        x = [1, 0, 0]
    elif row['engy_kind_cd'] == 12:
        x = [0, 1, 0]
    else:
        x = [0, 0, 1] # 지역냉난방

    ## 기저 소비량
    row['eb_electricity'] = x[0] * row['eb_qty']
    row['eb_gas'] = x[1] * row['eb_qty']
    row['eb_district'] = x[2] * row['eb_qty']
    row['eb_tot'] = row['eb_electricity']+row['eb_gas']+row['eb_district']

    ## 냉방 소비량
    row['ec_electricity'] = x[0] * row['ec_qty']
    row['ec_gas'] = x[1] * row['ec_qty']
    row['ec_district'] = x[2] * row['ec_qty']
    row['ec_tot'] = row['ec_electricity']+row['ec_gas']+row['ec_district']

    ## 난방 소비량
    row['eh_electricity'] = x[0] * row['eh_qty']
    row['eh_gas'] = x[1] * row['eh_qty']
    row['eh_district'] = x[2] * row['eh_qty']
    row['eh_tot'] = row['eh_electricity']+row['eh_gas']+row['eh_district']

    ## 합계 소비량
    row['tot_electricity'] = x[0] * row['tot_use_qty']
    row['tot_gas'] = x[1] * row['tot_use_qty']
    row['tot_district'] = x[2] * row['tot_use_qty']  # 합계 소비량
    row['tot_tot'] = row['tot_electricity']+row['tot_gas']+row['tot_district']

    return row

energy_m3 = energy_m3.apply(lambda row: divide_energy(row), axis=1)
print('에너지원별 열구분')
print("# of PK3 : ", energy_m3['mgm_bld_pk'].nunique())
print('# of data3 : ', energy_m3.shape[0])
print(' ')




## 기저분리별 연간 E 산정(표제부 기준 group)

aggregation_functions = {
    'eb_tot': 'sum', 'eb_electricity': 'sum', 'eb_gas': 'sum', 'eb_district': 'sum',
    'ec_tot': 'sum', 'ec_electricity': 'sum', 'ec_gas': 'sum', 'ec_district': 'sum',
    'eh_tot': 'sum', 'eh_electricity': 'sum', 'eh_gas': 'sum', 'eh_district': 'sum',
    'tot_tot': 'sum', 'tot_electricity': 'sum', 'tot_gas': 'sum', 'tot_district': 'sum'
}

def aggregate_energy(data):
    energy_agg = data.groupby(by=['mgm_bld_pk', 'engy_kind_cd', 'year_use', 'month_use']).agg(aggregation_functions).reset_index()
    return energy_agg

energy_m3_1 = aggregate_energy(energy_m3)

# 모든 월 목록
months = list(range(1, 13))

# 각 그룹에서 1월부터 12월까지 모두 있는지 확인하는 함수
def has_all_months(group):
    return set(months).issubset(set(group['month_use']))

# 그룹별로 필터링 적용
energy_m3_1 = energy_m3_1.groupby(by=['mgm_bld_pk', 'engy_kind_cd', 'year_use']).filter(has_all_months)

print('month = 1~12')
print("# of PK4 : ", energy_m3_1['mgm_bld_pk'].nunique())
print('# of data4 : ', energy_m3_1.shape[0])
print(' ')



## 연간 E 합계를 계산
energy_m4 = energy_m3_1.groupby(by=['mgm_bld_pk', 'year_use']).agg(aggregation_functions).reset_index()
print('기저분리별 연간 E 산정')
print("# of PK5 : ", energy_m4['mgm_bld_pk'].nunique())
print('# of data5 : ', energy_m4.shape[0])
print(' ')



## 전기E 0이 아닌 것 필터
energy_m5 = energy_m4[energy_m4['tot_electricity']>0] ## 기저/냉방/난방/합계 모두 0인 건 동일 126개
print('전기E 존재 필터')
print("# of PK6 : ", energy_m5['mgm_bld_pk'].nunique())
print('# of data6 : ', energy_m5.shape[0])
print(' ')



## 연도 구분
def year_division(A):
    B = A[A['year_use']==2018]
    C = A[A['year_use']==2019]
    return B,C

energy_m2018, energy_m2019 = year_division(energy_m5)
print('계량기별_2018')
print("# of PK7 : ", energy_m2018['mgm_bld_pk'].nunique())
print('# of data7 : ', energy_m2018.shape[0])
print(' ')
print('계량기별_2019')
print("# of PK7 : ", energy_m2019['mgm_bld_pk'].nunique())
print('# of data7 : ', energy_m2019.shape[0])
print(' ')


print(' ')
print("############################################################################################")


## 6_서울경기_의료기관_표제부 최종 필터본과 3_계량기별_사용량 최종 필터본 병합 ##################################################


m_total = pd.merge(left=bld_pk3, right=energy_m5, how='inner', on='mgm_bld_pk')
print('병합_계량기별_Total')
print("# of PK : ", m_total['mgm_bld_pk'].nunique())
print('# of data : ', m_total.shape[0])
print(' ')


m_2018 = pd.merge(left=bld_pk3, right=energy_m2018, how='inner', on='mgm_bld_pk')
m_2018.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저분리_계량기연간사용량(2018).xlsx")

print('병합_계량기별_2018')
print("# of PK : ", m_2018['mgm_bld_pk'].nunique())
print('# of data : ', m_2018.shape[0])
print(' ')

m_2019 = pd.merge(left=bld_pk3, right=energy_m2019, how='inner', on='mgm_bld_pk')
m_2019.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저분리_계량기연간사용량(2019).xlsx")

print('병합_계량기별_2019')
print("# of PK : ", m_2019['mgm_bld_pk'].nunique())
print('# of data : ', m_2019.shape[0])
print(' ')




# print(' ')
# print("4_에너지원별_사용량 ############################################################################################")
# ### 4_에너지원별_사용량 ############################################################################################
#
# ## 기본
# energy_e=df04
# def year_separation(A):
#     A['use_ym'] = pd.to_datetime(A['use_ym'], format='%Y%m')
#     A['year_use'] = A['use_ym'].dt.year           # 연도 구분
#     A['month_use'] = A['use_ym'].dt.month         # 월별 구분
#     return A
# energy_e=year_separation(energy_e)
#
# print('4_에너지원별_사용량')
# print("# of PK : ", energy_e['mgm_bld_pk'].nunique())
# print('# of data : ', energy_e.shape[0])
# print(' ')
#
# # PK_list = bld_pk3['mgm_bld_pk'].unique()
# # energy_e = energy_e[energy_e['mgm_bld_pk'].isin(PK_list)]
# # print('건축물대장 필터')
# # print("# of PK0 : ", energy_e['mgm_bld_pk'].nunique())
# # print('# of data0 : ', energy_e.shape[0])
# # print(' ')
#
# ## 일반건축물 추출
# energy_e1=energy_e
# energy_e1=energy_e1[energy_e1['regstr_gb_nm']=='일반']
# print('regstr_gb_nm ＝ 일반')
# print("# of PK1 : ", energy_e1['mgm_bld_pk'].nunique())
# print('# of data1 : ', energy_e1.shape[0])
# print(' ')
#
#
#
# ## 에너지원별 kWh기준 단위 통일(1: KWh 과 8: MJ 만 존재)
# def energy_conversion(row):
#     if row['unit_cd'] == 1:
#         x = 1
#     elif row['unit_cd'] == 2:
#         x = 42.7 * 1 / 3.6
#     elif row['unit_cd'] == 3:
#         x = 1 / 0.860 * 1000
#     elif row['unit_cd'] == 4:
#         x = 1000
#     elif row['unit_cd'] == 6:
#         x = 1 / 0.860
#     elif row['unit_cd'] == 8:
#         x = 1 / 3.6
#     else:
#         x = 63.4 * 1 / 3.6  # UNIT_CD = 14 #
#     row['eb_qty'] = x * row['eb_qty']            # 기저 소비량
#     row['ec_qty'] = x * row['ec_qty']            # 냉방 소비량
#     row['eh_qty'] = x * row['eh_qty']            # 난방 소비량
#     row['tot_use_qty'] = x * row['tot_use_qty']  # 합계 소비량
#     return row
#
# # ### energy conversion
# # - 1 Nm3 = 10.55 kWh
# # - 1 gcal = 42.7*1/3.6 kWh
# # - 1 MWh = 1000 kWh
# # - 1 Mcal = 1/0.860 kWh
# # - 1 MJ = 1/3.6 kWh
# # - 1 Nm3 = 63.4*1/3.6
#
# energy_e1 = energy_e1.apply(energy_conversion, axis=1)
#
#
#
# ## 12개월 모두 존재 데이터만 추출(11:전기, 12:가스, 13:지역냉난방)
# energy_e2=energy_e1
# def divide_energy(row):
#     if row['engy_kind_cd'] == 11:
#         x = [1, 0, 0]
#     elif row['engy_kind_cd'] == 12:
#         x = [0, 1, 0]
#     else:
#         x = [0, 0, 1] # 지역냉난방
#
#     ## 기저 소비량
#     row['eb_electricity'] = x[0] * row['eb_qty']
#     row['eb_gas'] = x[1] * row['eb_qty']
#     row['eb_district'] = x[2] * row['eb_qty']
#     row['eb_tot'] = row['eb_electricity']+row['eb_gas']+row['eb_district']
#
#     ## 냉방 소비량
#     row['ec_electricity'] = x[0] * row['ec_qty']
#     row['ec_gas'] = x[1] * row['ec_qty']
#     row['ec_district'] = x[2] * row['ec_qty']
#     row['ec_tot'] = row['ec_electricity']+row['ec_gas']+row['ec_district']
#
#     ## 난방 소비량
#     row['eh_electricity'] = x[0] * row['eh_qty']
#     row['eh_gas'] = x[1] * row['eh_qty']
#     row['eh_district'] = x[2] * row['eh_qty']
#     row['eh_tot'] = row['eh_electricity']+row['eh_gas']+row['eh_district']
#
#     ## 합계 소비량
#     row['tot_electricity'] = x[0] * row['tot_use_qty']
#     row['tot_gas'] = x[1] * row['tot_use_qty']
#     row['tot_district'] = x[2] * row['tot_use_qty']  # 합계 소비량
#     row['tot_tot'] = row['tot_electricity']+row['tot_gas']+row['tot_district']
#
#     return row
#
# energy_e2 = energy_e2.apply(lambda row: divide_energy(row), axis=1)
# print('에너지원별 열구분')
# print("# of PK3 : ", energy_e2['mgm_bld_pk'].nunique())
# print('# of data3 : ', energy_e2.shape[0])
# print(' ')
#
#
#
#
# ## 기저분리별 연간 E 산정(표제부 기준 group)
#
# aggregation_functions = {
#     'eb_tot': 'sum', 'eb_electricity': 'sum', 'eb_gas': 'sum', 'eb_district': 'sum',
#     'ec_tot': 'sum', 'ec_electricity': 'sum', 'ec_gas': 'sum', 'ec_district': 'sum',
#     'eh_tot': 'sum', 'eh_electricity': 'sum', 'eh_gas': 'sum', 'eh_district': 'sum',
#     'tot_tot': 'sum', 'tot_electricity': 'sum', 'tot_gas': 'sum', 'tot_district': 'sum'
# }
#
# def aggregate_energy(data):
#     energy_agg = data.groupby(by=['mgm_bld_pk', 'engy_kind_cd', 'year_use', 'month_use']).agg(aggregation_functions).reset_index()
#     return energy_agg
#
# energy_e2_1 = aggregate_energy(energy_e2)
#
# # 모든 월 목록
# months = list(range(1, 13))
#
# # 각 그룹에서 1월부터 12월까지 모두 있는지 확인하는 함수
# def has_all_months(group):
#     return set(months).issubset(set(group['month_use']))
#
# # 그룹별로 필터링 적용
# energy_e2_1 = energy_e2_1.groupby(by=['mgm_bld_pk', 'engy_kind_cd', 'year_use']).filter(has_all_months)
#
# print('month = 1~12')
# print("# of PK4 : ", energy_e2_1['mgm_bld_pk'].nunique())
# print('# of data4 : ', energy_e2_1.shape[0])
# print(' ')
#
#
#
# ## 연간 E 합계를 계산
# energy_e3 = energy_e2_1.groupby(by=['mgm_bld_pk', 'year_use']).agg(aggregation_functions).reset_index()
# print('기저분리별 연간 E 산정')
# print("# of PK5 : ", energy_e3['mgm_bld_pk'].nunique())
# print('# of data5 : ', energy_e3.shape[0])
# print(' ')
#
#
#
# ## 전기E 0이 아닌 것 필터
# energy_e4 = energy_e3[energy_e3['tot_electricity']>0] ## 기저/냉방/난방/합계 모두 0인 건 동일 126개
# print('전기E 존재 필터')
# print("# of PK6 : ", energy_e4['mgm_bld_pk'].nunique())
# print('# of data6 : ', energy_e4.shape[0])
# print(' ')
#
#
#
# ## 연도 구분
# def year_division(A):
#     B = A[A['year_use']==2018]
#     C = A[A['year_use']==2019]
#     return B,C
#
# energy_e2018, energy_e2019 = year_division(energy_e4)
# print('에너지원별_2018')
# print("# of PK7 : ", energy_e2018['mgm_bld_pk'].nunique())
# print('# of data7 : ', energy_e2018.shape[0])
# print(' ')
# energy_e2018.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저(2018).xlsx")
#
#
#
# print('에너지원별_2019')
# print("# of PK7 : ", energy_e2019['mgm_bld_pk'].nunique())
# print('# of data7 : ', energy_e2019.shape[0])
# print(' ')
# energy_e2019.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저(2019).xlsx")
#
#
# print(' ')
# print("#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####")
# ## 6_서울경기_의료기관_표제부 최종 필터본과 3_계량기별_사용량 최종 필터본 병합 ###################################################
#
# e_total = pd.merge(left=bld_pk3, right=energy_e4, how='inner', on='mgm_bld_pk')
# print('에너지원별')
# print("# of PK : ", e_total['mgm_bld_pk'].nunique())
# print('# of data : ', e_total.shape[0])
# print(' ')
#
#
#
#
# e_2018 = pd.merge(left=bld_pk3, right=energy_e2018, how='inner', on='mgm_bld_pk')
# e_2018.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저분리_에너지원연간사용량(2018).xlsx")
#
# print('병합_에너지원별_2018')
# print("# of PK : ", e_2018['mgm_bld_pk'].nunique())
# print('# of data : ', e_2018.shape[0])
# print(' ')
#
# e_2019 = pd.merge(left=bld_pk3, right=energy_e2019, how='inner', on='mgm_bld_pk')
# e_2019.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저분리_에너지원연간사용량(2019).xlsx")
#
# print('병합_에너지원별_2019')
# print("# of PK : ", e_2019['mgm_bld_pk'].nunique())
# print('# of data : ', e_2019.shape[0])
# print(' ')
#
#
# print(' ')
# print("############################################################################################")


## 의료시설 데이터 병합 ###################################################################################################
hos1=pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_데이터셋2..xlsx")
print('의료시설_시설정보')
print("# of PK : ", hos1['mgm_bld_pk'].nunique())
print('# of data : ', hos1.shape[0])
print(' ')

m_total = pd.merge(left=m_total, right=hos1, how='inner', on='mgm_bld_pk')
print('계량기별')
print("# of PK : ", m_total['mgm_bld_pk'].nunique())
print('# of data : ', m_total.shape[0])
print(' ')

m_18 = pd.merge(left=m_2018, right=hos1, how='inner', on='mgm_bld_pk')
print('의료시설_계량기별_2018')
print("# of PK : ", m_18['mgm_bld_pk'].nunique())
print('# of data : ', m_18.shape[0])
print(' ')

m_19 = pd.merge(left=m_2019, right=hos1, how='inner', on='mgm_bld_pk')
print('의료시설_계량기별_2019')
print("# of PK : ", m_19['mgm_bld_pk'].nunique())
print('# of data : ', m_19.shape[0])
print(' ')
print(' ')


# e_total = pd.merge(left=e_total, right=hos1, how='inner', on='mgm_bld_pk')
# print('에너지원별')
# print("# of PK : ", e_total['mgm_bld_pk'].nunique())
# print('# of data : ', e_total.shape[0])
# print(' ')
#
# e_18 = pd.merge(left=e_2018, right=hos1, how='inner', on='mgm_bld_pk')
# print('의료시설_에너지원별_2018')
# print("# of PK : ", e_18['mgm_bld_pk'].nunique())
# print('# of data : ', e_18.shape[0])
# print(' ')
#
# e_19 = pd.merge(left=e_2019, right=hos1, how='inner', on='mgm_bld_pk')
# print('의료시설_에너지원별_2019')
# print("# of PK : ", e_19['mgm_bld_pk'].nunique())
# print('# of data : ', e_19.shape[0])
# print(' ')
#
#





### 에너지사용량 excel 입력
Ene=pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\에너지사용량(편집).xlsx")

print('BEFORE : 에너지사용량')
print('# of PK1 : ', Ene['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', Ene.shape[0])                 ## of data1 : 1001706
print(' ')
Ene_df_2018 = Ene[Ene['year_use'] == 2018]


print('# of PK1 : ', Ene_df_2018['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', Ene_df_2018.shape[0])                 ## of data1 : 1001706
print(' ')



Ene_df_2019 = Ene[Ene['year_use'] == 2019]
print('# of PK1 : ', Ene_df_2019['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', Ene_df_2019.shape[0])                 ## of data1 : 1001706
print(' ')


Ene_df=pd.concat([Ene_df_2018, Ene_df_2019])
print('# of PK1 : ', Ene_df['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', Ene_df.shape[0])                 ## of data1 : 1001706
print(' ')


tod_2018 = pd.merge(left=Ene_df_2018, right=m_18, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
print('# of PK1 : ', tod_2018['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', tod_2018.shape[0])                 ## of data1 : 1001706
print(' ')

tod_2019 = pd.merge(left=Ene_df_2019, right=m_19, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
print('# of PK1 : ', tod_2019['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', tod_2019.shape[0])                 ## of data1 : 1001706
print(' ')

# tod_2018['오차율']= (tod_2018['USE_QTY_kWh']-tod_2018['tot_tot'])*100/tod_2018['USE_QTY_kWh']
# # tod_2018=tod_2018[tod_2018['오차율']!= 0]
#
# tod_2019['오차율']= (tod_2019['USE_QTY_kWh']-tod_2019['tot_tot'])*100/tod_2019['USE_QTY_kWh']
# # tod_2019=tod_2019[tod_2019['오차율']!= 0]
# A2018=tod_2018['매칭표제부PK'].nunique()
# A2019=tod_2019['매칭표제부PK'].nunique()
tod_2019.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\testq.xlsx")
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,5))
# plt.hist(tod_2018['오차율'], color = 'green', alpha = 0.8, bins=50, label=f'2018({A2018})')
# plt.hist(tod_2019['오차율'], color = 'blue', alpha = 0.8, bins=50, label=f'2019({A2019})')
#
# plt.legend()
# plt.xlabel('Error Rate (%)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
#















# print(' ')
# print("연도별 계량기vs에너지원 데이터 동일 여부 검토 #######################################################################")
#
# data18 = pd.merge(left=m_18, right=e_18, how='inner', on='mgm_bld_pk')
# print('계량기에너지원병합_2018')
# print("# of PK : ", data18['mgm_bld_pk'].nunique())
# print('# of data : ', data18.shape[0])
# print(' ')
#
# data19 = pd.merge(left=m_19, right=e_19, how='inner', on='mgm_bld_pk')
# print('계량기에너지원병합_2019')
# print("# of PK : ", data19['mgm_bld_pk'].nunique())
# print('# of data : ', data19.shape[0])
# print(' ')
#
# data18.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\2018.xlsx')
# data19.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\2019.xlsx')


print(' ')
print("############################################################################################")
# 데이터 로드 및 필터링
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
df_merge_result = pd.read_excel(file_path)
print("# of PK : ", df_merge_result['매칭표제부PK'].nunique())
print('# of data : ', df_merge_result.shape[0])

filtered_df = df_merge_result[df_merge_result['주용도(의료시설) 비율(%)'] >= 90]
print('90%이상')
print("# of PK : ", filtered_df['매칭표제부PK'].nunique())
print('# of data : ', filtered_df.shape[0])

filtered_df_2018 = filtered_df[filtered_df['year_use'] == 2018]
print('2018')
print("# of PK : ", filtered_df_2018['매칭표제부PK'].nunique())
print('# of data : ', filtered_df_2018.shape[0])

filtered_df_2019 = filtered_df[filtered_df['year_use'] == 2019]
print('2019')
print("# of PK : ", filtered_df_2019['매칭표제부PK'].nunique())
print('# of data : ', filtered_df_2019.shape[0])
print(' ')

filtered_1819 =pd.concat([filtered_df_2018, filtered_df_2019], axis=0)
print("# of PK : ", filtered_df['매칭표제부PK'].nunique())
print('# of data : ', filtered_df.shape[0])
print(' ')

print("############################################################################################")
merge_m = pd.merge(left=filtered_1819, right=m_total, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
print("# of PK : ", merge_m['매칭표제부PK'].nunique())
print('# of data : ', merge_m.shape[0])
print(' ')

# merge_e = pd.merge(left=filtered_1819, right=e_total, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
# print("# of PK : ", merge_e['매칭표제부PK'].nunique())
# print('# of data : ', merge_e.shape[0])
# print(' ')

merge_m18 = pd.merge(left=filtered_df_2018, right=m_18, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
merge_m19 = pd.merge(left=filtered_df_2019, right=m_19, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
# merge_e18 = pd.merge(left=filtered_df_2018, right=e_18, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
# merge_e19 = pd.merge(left=filtered_df_2019, right=e_19, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
print('m2018')
print("# of PK : ", merge_m18['매칭표제부PK'].nunique())
print('# of data : ', merge_m18.shape[0])
print('m2019')
print("# of PK : ", merge_m19['매칭표제부PK'].nunique())
print('# of data : ', merge_m19.shape[0])
# print('e2018')
# print("# of PK : ", merge_e18['매칭표제부PK'].nunique())
# print('# of data : ', merge_e18.shape[0])
# print('e2019')
# print("# of PK : ", merge_e19['매칭표제부PK'].nunique())
# print('# of data : ', merge_e19.shape[0])




error_rate_18 = (merge_m18['USE_QTY_kWh'] - merge_m18['tot_tot']) * 100 / merge_m18['USE_QTY_kWh']
error_rate_19 = (merge_m19['USE_QTY_kWh'] - merge_m19['tot_tot']) * 100 / merge_m19['USE_QTY_kWh']

print(error_rate_18.mean())
print(error_rate_19.mean())

merge_m18 = pd.concat([merge_m18, error_rate_18.rename('오차율')], axis=1)
merge_m19 = pd.concat([merge_m19, error_rate_19.rename('오차율')], axis=1)
merge_m18.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토m2018.xlsx')
merge_m19.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토m2019.xlsx')
# merge_e18.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토e2018.xlsx')
# merge_e19.to_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토e2019.xlsx')


merge_m18=pd.read_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토m2018.xlsx')
merge_m19=pd.read_excel(r'C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\검토m2019.xlsx')
print('m2018')
print("# of PK : ", merge_m18['매칭표제부PK'].nunique())
print('# of data : ', merge_m18.shape[0])
print('m2019')
print("# of PK : ", merge_m19['매칭표제부PK'].nunique())
print('# of data : ', merge_m19.shape[0])
A2018=merge_m18['매칭표제부PK'].nunique()
A2019=merge_m19['매칭표제부PK'].nunique()


# 데이터프레임을 복사하여 조각화를 방지
mer18 = merge_m18.copy()
mer19 = merge_m19.copy()
mer18=mer18[mer18['오차율']!= 0]
mer19=mer19[mer19['오차율']!= 0]
print(mer18['매칭표제부PK'].nunique())
print(mer19['매칭표제부PK'].nunique())

import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.hist(mer18['오차율'], color='green', alpha=0.8, bins=50, label=f'2018({A2018})')
plt.hist(mer19['오차율'], color='blue', alpha=0.8, bins=50, label=f'2019({A2019})')
# 여러 데이터셋을 한 번에 그리면서 색상을 리스트로 지정
plt.legend()
plt.xlabel('Error Rate (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# testcbrj