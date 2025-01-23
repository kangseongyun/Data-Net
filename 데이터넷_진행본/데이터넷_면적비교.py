import os
import pandas as pd
from matplotlib import pyplot as plt

dir_path= r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename="데이터넷_1_01_시설정보.xlsx"
# filename="데이터넷_1_02_세부정보.xlsx"
# filename="데이터넷_1_03_진료과목정보.xlsx"
# filename="데이터넷_1_05_의료장비정보.xlsx"
# filename="데이터넷_1_06_식대가산정보.xlsx"
# filename="데이터넷_1_07_간호등급정보.xlsx"
# filename="데이터넷_1_08_특수진료정보.xlsx"


file_path = os.path.join(dir_path,filename)
df_merge_result=pd.read_excel(file_path)

print("# of PK : ", df_merge_result['매칭표제부PK'].nunique())  # of PK :    1429 / 786  / 1550 / 1055 / 1470 / 1465 / 353
print('# of data : ', df_merge_result.shape[0])               # of data :  6046 / 3406 / 6547 / 4449 / 6200 / 6188 / 1448
print(' ')

df_merge_result = df_merge_result[df_merge_result['주용도(의료시설) 비율(%)'] >= 90]
print("# of PK : ", df_merge_result['매칭표제부PK'].nunique())  # of PK :    1429 / 786  / 1550 / 1055 / 1470 / 1465 / 353
print('# of data : ', df_merge_result.shape[0])               # of data :  6046 / 3406 / 6547 / 4449 / 6200 / 6188 / 1448
print(' ')

df_merge_result = df_merge_result.drop_duplicates(subset='매칭표제부PK')
df_merge_result=df_merge_result[df_merge_result['면적 합계']!= df_merge_result['연면적(㎡)']]
# df_merge_result=df_merge_result[df_merge_result['면적 합계']==df_merge_result['총동연면적(㎡)']]
df_merge_result['오차율']=(df_merge_result['면적 합계']-df_merge_result['연면적(㎡)'])*100/df_merge_result['면적 합계']

print('동일성 검사')
print("# of PK : ", df_merge_result['매칭표제부PK'].nunique())  # of PK :    1429 / 786  / 1550 / 1055 / 1470 / 1465 / 353
print('# of data : ', df_merge_result.shape[0])               # of data :  6046 / 3406 / 6547 / 4449 / 6200 / 6188 / 1448
print(' ')

max_rate=df_merge_result['오차율'].max()
min_rate=df_merge_result['오차율'].min()
print(max_rate.round(1))
print(min_rate.round(1))
print(df_merge_result['오차율'])
print(df_merge_result['오차율'].mean())

plt.hist(df_merge_result['오차율'], color = 'green', alpha = 0.4, bins=50, label='bins=50')
plt.legend()
plt.xlabel('Error Rate (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


df_merge=df_merge_result[['매칭표제부PK','면적 합계','연면적(㎡)']]
df_merge=df_merge.drop_duplicates()
q2018=df_merge.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본\데이터넷_면적검토.xlsx")# of PK :  1139







# ### 데이터셋1, 2, 3 병합 연도별로 excel Data로 export ########################################################################
#
#
# ## 건축물대장과 에너지사용량 동일 경로 입력
# base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"
#
#
# ### 에너지사용량 excel 입력
# filenames_energy = ["데이터넷_의료시설(에너지사용량_2018년).xlsx",
#                     "데이터넷_의료시설(에너지사용량_2019년).xlsx",
#                     "데이터넷_의료시설(에너지사용량_2020년).xlsx",
#                     "데이터넷_의료시설(에너지사용량_2021년).xlsx",
#                     "데이터넷_의료시설(에너지사용량_2022년).xlsx"]
# sheetnames_energy = ['표제부_에너지사용량_계량기_2018년',
#                      '표제부_에너지사용량_계량기_2019년',
#                      '표제부_에너지사용량_계량기_2020년',
#                      '표제부_에너지사용량_계량기_2021년',
#                      '표제부_에너지사용량_계량기_2022년']
#
#
#
# ### step 1. 데이터 호출, 기본 설정 #######################################################################################
#
# df_en = pd.DataFrame()
# for i in range(0, len(filenames_energy)):
#     file_path = os.path.join(base_dir, filenames_energy[i])
#     df = pd.read_excel(file_path, sheet_name=sheetnames_energy[i]) # 엑셀 데이터 호출
#     df_en = pd.concat([df_en, df], ignore_index=True)         # 엑셀 데이터 병합
#
# df_en.rename(columns={'매칭총괄표제부PK': '매칭표제부PK'}, inplace=True) ## 기존 매칭총괄표제부PK를 매칭표제부PK로 정정
# df_en['사용년월'] = pd.to_datetime(df_en['사용년월'], format='%Y%m')
# df_en['year_use'] = df_en['사용년월'].dt.year           # 연도 구분
# df_en['month_use'] = df_en['사용년월'].dt.month         # 월별 구분은 추후에 진행할 예정
#
# print('BEFORE : 에너지사용량')
# print('# of PK1 : ', df_en['매칭표제부PK'].nunique())    ## of PK１ : 4104
# print('# of data1 : ', df_en.shape[0])                 ## of data1 : 1001706
# print(' ')
#
#
#
# ### step 3. 사용량 ≠ NaN & 사용량 > 0 ###################################################################################
#
# df_en = df_en[~df_en['사용량'].isna()]
# df_en['사용량'] = df_en['사용량'].astype(float)
# df_en = df_en[df_en['사용량'] > 0]
#
# print('사용량 ≠ NaN & 사용량 > 0')
# print('# of PK3 : ', df_en['매칭표제부PK'].nunique())    ## of PK3 : 3013
# print('# of data3 : ',len(df_en))                      ## of data3 :  693086
# print(' ')
#
#
#
# ### step 4. 에너지 종류 표기(전기/도시가스/지역난방) #######################################################################
#
# df_en = df_en[~df_en['에너지공급기관코드'].isna()]       ## '에너지공급기관코드'열에 모두 value가 존재하므로 의미가 없음.
# df_en['에너지공급기관코드'] = df_en['에너지공급기관코드'].astype(str)
#
#
# filename = filenames_energy[0]                        ## 일단 "데이터넷_의료시설(에너지사용량_2018년).xlsx"을 기준으로 실행
# file_path = os.path.join(base_dir, filename)
# df_list = pd.read_excel(file_path, sheet_name='에너지용도코드', dtype=str) ## 시트를 '에너지용도코드'로 실행
#
#
# mapping_data1 = df_list[['기관코드', '에너지종류']]
# df_en['에너지종류'] = df_en.apply(lambda row: mapping_data1[mapping_data1['기관코드'] == row['에너지공급기관코드']]['에너지종류'].values[0] if row['에너지공급기관코드'] in mapping_data1['기관코드'].values else '', axis=1)
#
#
#
# ### step 5. 단위코드 열을 kWh기준 단위 환산 ###############################################################################
#
#
# df_en = df_en[~df_en['단위코드'].isna()] ## '단위코드'열에 모두 value가 존재하므로 의미가 없음.
#
# print('에너지공급기관코드 ≠ NaN & 단위코드 ≠ NaN')
# print('# of PK４ : ', df_en['매칭표제부PK'].nunique())    ## of PK4 : 3013
# print('# of data４ : ',len(df_en))                      ## of data4 :  693086
# print(' ')
#
#
# def energy_conversion(row):
#     if row['단위코드'] == '01':
#         x = 1
#     elif row['단위코드'] == '02':
#         x = 42.7 * 1 / 3.6
#     elif row['단위코드'] == '03':
#         x = 1 / 0.860 * 1000
#     elif row['단위코드'] == '04':
#         x = 1000
#     elif row['단위코드'] == '06':
#         x = 1 / 0.860
#     elif row['단위코드'] == '08':
#         x = 1 / 3.6
#     else:
#         x = 63.4 * 1 / 3.6  # UNIT_CD = 14 #
#     return x * row['사용량']
#
# # ### energy conversion
# # - 1 Nm3 = 10.55 kWh
# # - 1 gcal = 42.7*1/3.6 kWh
# # - 1 MWh = 1000 kWh
# # - 1 Mcal = 1/0.860 kWh
# # - 1 MJ = 1/3.6 kWh
# # - 1 Nm3 = 63.4*1/3.6
#
# df_en['USE_QTY_kWh'] = df_en.apply(lambda row: energy_conversion(row), axis=1)
#
#
# ### step 6. 에너지 종류에 따른 열 구분 & 연간총에너지사용량 및 갯수 산정 ########################################################
#
# def divide_energy(row):
#     if row['에너지종류'] == '전기':
#         x = [1, 0, 0]
#     elif row['에너지종류'] == '도시가스':
#         x = [0, 1, 0]
#     else:
#         x = [0, 0, 1] # 지역난방
#     import numpy as np
#     return np.array(x) * row['USE_QTY_kWh']
#
# df_en[['electricity_kWh', 'gas_kWh', 'districtheat_kWh']] = df_en.apply(lambda row: divide_energy(row), axis=1, result_type='expand')
#
#
# df_1 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])[['USE_QTY_kWh','electricity_kWh','gas_kWh','districtheat_kWh']].sum() ## 연간 총 에너지사용량 산출
# df_2 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])['USE_QTY_kWh'].count().to_frame() ## 매칭표제부PK, 에너지종류 및 연도별로 존재하는 갯수 산출
# df_2.columns = ['count_energy']
#
# df_en_annual = pd.concat([df_1, df_2], axis = 1).sort_index(axis=1) ## df_1, df_2 병합
# df_en_annual.reset_index(inplace=True)
#
# print('# of PK1 : ', df_en_annual['매칭표제부PK'].nunique())    ## of PK１ : 4104
# print('# of data1 : ', df_en_annual.shape[0])                 ## of data1 : 1001706
# print(' ')
#
# ### step 7. 연간 전기E의 Data가 존재하는 매칭표제부PK만 도출 #################################################################
#
# df_result = pd.DataFrame()
# for t in df_en_annual['에너지종류'].unique():
#     df_ = df_en_annual[df_en_annual['에너지종류']==t]
#     if t == '전기':
#         df_ = df_[df_['count_energy']%12==0]    ## 계량기가 두개이상일 경우 여기서 각각 개별적으로 연간데이터가 모두 존재할 때는 12로 나누어져야 하기 때문에 나머지가 0이 되어야 함.
#     else:
#         df_ = df_   # 전기를 제외한 나머지에 대해서는 굳이 연간 데이터 모두 존재하지 않아도 넘어감.
#     df_result = pd.concat([df_result,df_])
#
# print('# of PK5 : ', df_result['매칭표제부PK'].nunique())       # of PK5 : 2950
# print('# of data5 : ',len(df_result))                          # of data5 :  21112
# print(' ')
#
# # 매칭표제부별로 전기 data 갯수가 0개인 경우도 나머지가 0이므로 이를 제거해야 함.
# electric_PKs = df_result[df_result['에너지종류'] == '전기']['매칭표제부PK'].unique()
# df_en_annual = df_result[df_result['매칭표제부PK'].isin(electric_PKs)]
#
# print('count(전기) = 12')
# print('# of PK5 : ', df_en_annual['매칭표제부PK'].nunique())     # of PK5 :  2875
# print('# of data5 : ',len(df_en_annual))                        # of data5 :  20777
# print(' ')
#
#
# ### step 8. 연도별로 전기가 모두 존재하는 PK의 연도만 출력 ####################################################################
#
# # 각 PK를 기준으로 그룹화
# valid_dfs = []
#
# # 각 PK를 기준으로 그룹화
# grouped = df_en_annual.groupby('매칭표제부PK')
#
# # 그룹 내에서 전기와 도시가스가 모두 있는 연도 찾기
# for _, group in grouped:
#     # 전기가 존재하는 연도 찾기
#     electricity_years = group[group['에너지종류'] == '전기']['year_use'].unique()
#     is_valid_year = group['year_use'].isin(electricity_years)
#     valid_dfs.append(group[is_valid_year])
#
# # 유효한 연도에 해당하는 행만 포함하는 최종 데이터 프레임 생성
# df_en_annual1 = pd.concat(valid_dfs)
#
#
# print('# of PK6 : ', df_en_annual1['매칭표제부PK'].nunique())      ## of PK5 :  1552
# print('# of data6 : ',len(df_en_annual1))                        ## of data5 :  12422
# print(' ')
#
# df_en_annual1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\sample1.xlsx")
#
# ### step 9. 데이터프레임 정리/연간총에너지 산정 ##############################################################################
# df_en_annual3 = df_en_annual1[['매칭표제부PK', '에너지종류', 'year_use', 'USE_QTY_kWh','electricity_kWh','gas_kWh','districtheat_kWh']]
#
# df_en_annual1 = df_en_annual3.groupby(['매칭표제부PK', 'year_use']).agg({
#     'USE_QTY_kWh': 'sum',
#     'electricity_kWh': 'sum',
#     'gas_kWh': 'sum',
#     'districtheat_kWh': 'sum'
# }).reset_index()
#
# print('df_en_annual : 최종_연간 총 에너지사용량')
# print("# of PK7 : ",df_en_annual1['매칭표제부PK'].nunique())      # of PK2 :  2875
# print('# of data7 : ', df_en_annual1.shape[0])                  # of data2 :  20777
# print(' ')
#
#
# df_en_annual2018 = df_en_annual1[df_en_annual1['year_use']==2018]
# df_en_annual2019 = df_en_annual1[df_en_annual1['year_use']==2019]
#
#
#
#
# Ene2018= pd.read_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저(2018).xlsx")
# Ene2019= pd.read_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\병합본\기저(2019).xlsx")
#
#
# print('# of PK1 : ', Ene2018['mgm_bld_pk'].nunique())    ## of PK１ : 4104
# print('# of data1 : ', Ene2018.shape[0])                 ## of data1 : 1001706
# print(' ')
#
# print('# of PK1 : ', Ene2019['mgm_bld_pk'].nunique())    ## of PK１ : 4104
# print('# of data1 : ', Ene2019.shape[0])                 ## of data1 : 1001706
# print(' ')
#
#
# tot_ene=pd.concat([Ene2018,Ene2019])
# print('# of PK1 : ', tot_ene['mgm_bld_pk'].nunique())    ## of PK１ : 4104
# print('# of data1 : ', tot_ene.shape[0])                 ## of data1 : 1001706
# print(' ')
#
#
#
# e_18 = pd.merge(left=df_en_annual2018, right=Ene2018, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
# print('의료시설_에너지원별_2018')
# print("# of PK : ", e_18['매칭표제부PK'].nunique())
# print('# of data : ', e_18.shape[0])
# print(' ')
#
# e_19 = pd.merge(left=df_en_annual2019, right=Ene2019, how='inner', left_on='매칭표제부PK', right_on='mgm_bld_pk')
# print('의료시설_에너지원별_2019')
# print("# of PK : ", e_19['매칭표제부PK'].nunique())
# print('# of data : ', e_19.shape[0])
# print(' ')
#
# e_18.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\2018_A.xlsx")
# e_19.to_excel(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일\대충파일\2019_A.xlsx")
#
# # print(tod)
# #
# e_18['오차율']= (e_18['USE_QTY_kWh']-e_18['tot_tot'])*100/e_18['USE_QTY_kWh']
# e_18=e_18[e_18['오차율']!= 0]
# e_19['오차율']= (e_19['USE_QTY_kWh']-e_19['tot_tot'])*100/e_19['USE_QTY_kWh']
# e_19=e_19[e_19['오차율']!= 0]
# print(e_19.mean())
#
# plt.hist(e_18['오차율'], color = 'green', alpha = 0.4, bins=50, label='2018(bins=50)')
# plt.hist(e_19['오차율'], color = 'blue', alpha = 0.4, bins=50, label='2019(bins=50)')
#
# plt.legend()
# plt.xlabel('Error Rate (%)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()