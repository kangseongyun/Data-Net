import os
import numpy as np
import pandas as pd
base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

filenames_energy = ["데이터넷_의료시설(에너지사용량_2022년).xlsx"]
sheetnames_energy = ['표제부_에너지사용량_계량기_2022년']


Final = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본\데이터넷_1_01_시설정보.xlsx")


file_path = os.path.join(base_dir, filenames_energy[0])

# Load the energy data from the specified sheet
df_list = pd.read_excel(file_path, sheet_name=sheetnames_energy[0])
df_list.rename(columns={'매칭총괄표제부PK': '매칭표제부PK'}, inplace=True) ## 기존 매칭총괄표제부PK를 매칭표제부PK로 정정
df_list['사용년월'] = pd.to_datetime(df_list['사용년월'], format='%Y%m')
df_list['year_use'] = df_list['사용년월'].dt.year           # 연도 구분
df_list['month_use'] = df_list['사용년월'].dt.month         # 월별 구분은 추후에 진행할 예정



Step1=pd.DataFrame()
Step1['매칭표제부PK']=Final['매칭표제부PK']






### step 4. 에너지 종류 표기(전기/도시가스/지역난방) #######################################################################

df_en = df_list[~df_list['에너지공급기관코드'].isna()]       ## '에너지공급기관코드'열에 모두 value가 존재하므로 의미가 없음.
df_en['에너지공급기관코드'] = df_en['에너지공급기관코드'].astype(str)


filename = filenames_energy[0]                        ## 일단 "데이터넷_의료시설(에너지사용량_2018년).xlsx"을 기준으로 실행
file_path = os.path.join(base_dir, filename)
df_list2 = pd.read_excel(file_path, sheet_name='에너지용도코드', dtype=str) ## 시트를 '에너지용도코드'로 실행


mapping_data1 = df_list2[['기관코드', '에너지종류']]
df_en['에너지종류'] = df_en.apply(lambda row: mapping_data1[mapping_data1['기관코드'] == row['에너지공급기관코드']]['에너지종류'].values[0] if row['에너지공급기관코드'] in mapping_data1['기관코드'].values else '', axis=1)


print(df_en['사용량'])
def energy_conversion(row):
    if row['단위코드'] == '01':
        x = 1
    elif row['단위코드'] == '02':
        x = 42.7 * 1 / 3.6
    elif row['단위코드'] == '03':
        x = 1 / 0.860 * 1000
    elif row['단위코드'] == '04':
        x = 1000
    elif row['단위코드'] == '06':
        x = 1 / 0.860
    elif row['단위코드'] == '08':
        x = 1 / 3.6
    else:
        x = 63.4 * 1 / 3.6  # UNIT_CD = 14 #
    return x * row['사용량']

# ### energy conversion
# - 1 Nm3 = 10.55 kWh
# - 1 gcal = 42.7*1/3.6 kWh
# - 1 MWh = 1000 kWh
# - 1 Mcal = 1/0.860 kWh
# - 1 MJ = 1/3.6 kWh
# - 1 Nm3 = 63.4*1/3.6
df_en['단위코드'] = df_en['단위코드'].apply(lambda x: '{:02d}'.format(int(x)) if isinstance(x, (int, float)) else x)
df_en['사용량'] = df_en.apply(lambda row: energy_conversion(row), axis=1)


hos_merge1 = pd.merge(left=Step1, right=df_en, how='inner',on=['매칭표제부PK'])


hos_merge1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\t.xlsx")

print("# of PK5 : ", hos_merge1['매칭표제부PK'].nunique())
print('# of data5 : ',len(hos_merge1))
print(' ')
hos_merge1=hos_merge1[hos_merge1['에너지종류']=='전기']
# hos_merge1=hos_merge1[hos_merge1['에너지종류']=='도시가스']
# hos_merge1=hos_merge1[hos_merge1['에너지종류']=='지역난방']


hos_grouped = hos_merge1.groupby(['매칭표제부PK', 'month_use'])['사용량'].sum().reset_index()
print("# of PK5 : ", hos_grouped['매칭표제부PK'].nunique())
print('# of data5 : ',len(hos_grouped))
print(' ')


hos_grouped= hos_grouped.pivot_table(index='매칭표제부PK',columns='month_use',values='사용량').reset_index()
print("# of PK5 : ", hos_grouped['매칭표제부PK'].nunique())
print('# of data5 : ',len(hos_grouped))
print(' ')
hos_grouped.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\t1.xlsx")

hos_merge1 = pd.merge(left=Final, right=hos_grouped, how='inner',on=['매칭표제부PK'])
print("# of PK5 : ", hos_merge1['매칭표제부PK'].nunique())
print('# of data5 : ',len(hos_merge1))
print(' ')



hos_merge1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_월별에너지사용량(전기).xlsx")
# hos_merge1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_월별에너지사용량(도시가스).xlsx")

# hos_merge1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_월별에너지사용량(지역난방).xlsx")

# 데이터프레임에서 모든 값이 동일한 행을 찾고 제거하는 코드
cols = [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12]  # 1~12열에 해당하는 열 이름 리스트
hos_merge = hos_merge1[~hos_merge1[cols].apply(lambda row: row.nunique() == 1, axis=1)]
print("# of PK5 : ", hos_merge['매칭표제부PK'].nunique())
print('# of data5 : ',len(hos_merge))
print(' ')
# hos_merge = hos_merge1[hos_merge1[cols].apply(lambda row: row.nunique() == 1, axis=1)]

# 연속으로 5번 동일한 값이 있는지 확인하는 함수
def has_five_consecutive_duplicates(row):
    return any(row.rolling(window=5).apply(lambda x: len(set(x)) == 1).dropna() == 1)

# 연속 5번 동일한 값이 있는 행 추출
hos_merge = hos_merge1[hos_merge1[cols].apply(has_five_consecutive_duplicates, axis=1)]





hos_merge.to_excel(r"C:\Users\user\Desktop\삭제용(전기).xlsx")
# hos_merge.to_excel(r"C:\Users\user\Desktop\삭제용(도시가스).xlsx")
# hos_merge.to_excel(r"C:\Users\user\Desktop\삭제용(지역난방).xlsx")