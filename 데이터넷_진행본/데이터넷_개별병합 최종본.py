import os
import numpy as np
import pandas as pd

##################################### 데이터셋1. 건축물대장 + 에너지사용량 ####################################################################################################

### 데이터셋1. Data input #####################################################################################################################################################

## 건축물대장과 에너지사용량 동일 경로 입력
base_dir = r"C:\Lab\01_Research\01_DataNet\데이터무덤\데이터넷_의료시설(건축물대장,에너지사용량)"

## 건축물대장 excel 입력
filenames_bldg="데이터넷_의료시설(건축물대장).xlsx"
sheetnames_bldg="건축물대장(표제부)"


### 에너지사용량 excel 입력
filenames_energy = [
                    # "데이터넷_의료시설(에너지사용량_2018년).xlsx",
                    # "데이터넷_의료시설(에너지사용량_2019년).xlsx",
                    # "데이터넷_의료시설(에너지사용량_2020년).xlsx",
                    # "데이터넷_의료시설(에너지사용량_2021년).xlsx",
                    "데이터넷_의료시설(에너지사용량_2022년).xlsx"]
sheetnames_energy = [
                     # '표제부_에너지사용량_계량기_2018년',
                     # '표제부_에너지사용량_계량기_2019년',
                     # '표제부_에너지사용량_계량기_2020년',
                     # '표제부_에너지사용량_계량기_2021년',
                     '표제부_에너지사용량_계량기_2022년']




############################################# 데이터셋1. 건축물대장 ##########################################################################################################
print('데이터셋1. 건축물대장 #############################################################################################')
print(' ')


### step 1. 건축물대장 데이터 호출 ########################################################################################

file_path = os.path.join(base_dir, filenames_bldg)
df_bldg1 = pd.read_excel(file_path, sheet_name=sheetnames_bldg)

print('Before : 건축물대장')
print('# of PK1 : ', len(df_bldg1))                    # of PK1 :  7581
print(' ')


### step 2. 데이터가 존재하는 열만 추출 ###################################################################################

columns_to_drop = df_bldg1.columns[78:]
df_bldg1 = df_bldg1.drop(columns=columns_to_drop)



### step 3. 매칭표제부PK ≠ NaN & 매칭총괄표제부PK = NaN ##################################################################

df_bldg1 = df_bldg1[df_bldg1['매칭총괄표제부PK'].isna()]
df_bldg1 = df_bldg1[df_bldg1['매칭표제부PK'].notna()]

print('매칭표제부PK ≠ NaN & 매칭총괄표제부PK = NaN')
print('# of PK2 : ',df_bldg1['매칭표제부PK'].nunique())     # of PK2 :  4318
print(' ')



### step 4. 대장구분코드명 = 일반 ########################################################################################

df_bldg1 = df_bldg1[df_bldg1['대장구분코드명']=='일반']

print('대장구분코드명 = 일반')
print('# of PK3 : ',df_bldg1['매칭표제부PK'].nunique())     # of PK3 :  3911
print(' ')



### step 5. 용적률산정연면적>0 & 연면적>0 #################################################################################

### dtype을 str로 했기 때문에 float로 수정
# df_bldg1['용적률산정연면적(㎡)'] = df_bldg1['용적률산정연면적(㎡)'].astype(float)
# df_bldg1['연면적(㎡)'] = df_bldg1['연면적(㎡)'].astype(float)

df_bldg1 = df_bldg1[(~df_bldg1['용적률산정연면적(㎡)'].isna())&(~df_bldg1['연면적(㎡)'].isna())]
df_bldg1 = df_bldg1[(df_bldg1['용적률산정연면적(㎡)']>0)&(df_bldg1['연면적(㎡)']>0)&(df_bldg1['대지면적(㎡)']>0)&(df_bldg1['지상층수']>0)
                    & (df_bldg1['건폐율(%)'] > 0)& (df_bldg1['건축면적(㎡)'] > 0) & (df_bldg1['용적률(%)'] > 0) ]

df_bldg1['층수'] = df_bldg1['지상층수']+df_bldg1['지하층수']
df_bldg1['승강기수'] = df_bldg1['비상용승강기수'] + df_bldg1['승용승강기수']

print('용적률산정연면적>0 & 연면적>0')
print('# of PK4 : ',df_bldg1['매칭표제부PK'].nunique())     # of PK4 :  3758
print(' ')
print(' ')


# ### step 6. 연간총에너지 사용량 excel Data로 export #######################################################################
# df_bldg1.to_excel(r'C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_데이터넷_건축물대장(최종).xlsx')





############################################# 데이터셋 1. 에너지사용량 ##########################################################################################################
print('데이터셋1. 에너지사용량 ###########################################################################################')
print(' ')


### step 1. 데이터 호출, 기본 설정 #######################################################################################

df_en = pd.DataFrame()
for i in range(0, len(filenames_energy)):
    file_path = os.path.join(base_dir, filenames_energy[i])
    df = pd.read_excel(file_path, sheet_name=sheetnames_energy[i]) # 엑셀 데이터 호출
    df_en = pd.concat([df_en, df], ignore_index=True)         # 엑셀 데이터 병합

df_en.rename(columns={'매칭총괄표제부PK': '매칭표제부PK'}, inplace=True) ## 기존 매칭총괄표제부PK를 매칭표제부PK로 정정
df_en['사용년월'] = pd.to_datetime(df_en['사용년월'], format='%Y%m')
df_en['year_use'] = df_en['사용년월'].dt.year           # 연도 구분
df_en['month_use'] = df_en['사용년월'].dt.month         # 월별 구분은 추후에 진행할 예정

print('BEFORE : 에너지사용량')
print('# of PK1 : ', df_en['매칭표제부PK'].nunique())    ## of PK１ : 4104
print('# of data1 : ', df_en.shape[0])                 ## of data1 : 1001706
print(' ')



### step 2. 필터된 건축물대장에 있는 매칭표제부PK만 사용 ####################################################################

# PK_list = df_bldg1['매칭표제부PK'].unique()              ## 전처리된 건축물대장 PK list입력
# df_en = df_en[df_en['매칭표제부PK'].isin(PK_list)]
#
# print('전처리된 건축물대장 PK list로 필터')
# print('# of PK2 : ', df_en['매칭표제부PK'].nunique())    ## of PK2 : 3023
# print('# of data2 : ', df_en.shape[0])                 ## of data2 : 742508
# print(' ')



### step 3. 사용량 ≠ NaN & 사용량 > 0 ###################################################################################

df_en = df_en[~df_en['사용량'].isna()]
df_en['사용량'] = df_en['사용량'].astype(float)
df_en = df_en[df_en['사용량'] > 0]

print('사용량 ≠ NaN & 사용량 > 0')
print('# of PK3 : ', df_en['매칭표제부PK'].nunique())    ## of PK3 : 3013
print('# of data3 : ',len(df_en))                      ## of data3 :  693086
print(' ')



### step 4. 에너지 종류 표기(전기/도시가스/지역난방) #######################################################################

df_en = df_en[~df_en['에너지공급기관코드'].isna()]       ## '에너지공급기관코드'열에 모두 value가 존재하므로 의미가 없음.
df_en['에너지공급기관코드'] = df_en['에너지공급기관코드'].astype(str)


filename = filenames_energy[0]                        ## 일단 "데이터넷_의료시설(에너지사용량_2018년).xlsx"을 기준으로 실행
file_path = os.path.join(base_dir, filename)
df_list = pd.read_excel(file_path, sheet_name='에너지용도코드', dtype=str) ## 시트를 '에너지용도코드'로 실행


mapping_data1 = df_list[['기관코드', '에너지종류']]
df_en['에너지종류'] = df_en.apply(lambda row: mapping_data1[mapping_data1['기관코드'] == row['에너지공급기관코드']]['에너지종류'].values[0] if row['에너지공급기관코드'] in mapping_data1['기관코드'].values else '', axis=1)



### step 5. 단위코드 열을 kWh기준 단위 환산 ###############################################################################


df_en = df_en[~df_en['단위코드'].isna()] ## '단위코드'열에 모두 value가 존재하므로 의미가 없음.
print('에너지공급기관코드 ≠ NaN & 단위코드 ≠ NaN')
print('# of PK４ : ', df_en['매칭표제부PK'].nunique())    ## of PK4 : 3013
print('# of data４ : ',len(df_en))                      ## of data4 :  693086
print(' ')


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

df_en['USE_QTY_kWh'] = df_en.apply(lambda row: energy_conversion(row), axis=1)
# df_en.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\t.xlsx")



### step 6. 에너지 종류에 따른 열 구분 & 연간총에너지사용량 및 갯수 산정 ########################################################

def divide_energy(row):
    if row['에너지종류'] == '전기':
        x = [1, 0, 0]
    elif row['에너지종류'] == '도시가스':
        x = [0, 1, 0]
    else:
        x = [0, 0, 1] # 지역난방
    return np.array(x) * row['USE_QTY_kWh']

df_en[['electricity_kWh', 'gas_kWh', 'districtheat_kWh']] = df_en.apply(lambda row: divide_energy(row), axis=1, result_type='expand')


df_1 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])[['USE_QTY_kWh','electricity_kWh','gas_kWh','districtheat_kWh']].sum() ## 연간 총 에너지사용량 산출
df_2 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])['USE_QTY_kWh'].count().to_frame() ## 매칭표제부PK, 에너지종류 및 연도별로 존재하는 갯수 산출
df_2.columns = ['count_energy']

df_en_annual = pd.concat([df_1, df_2], axis = 1).sort_index(axis=1) ## df_1, df_2 병합
df_en_annual.reset_index(inplace=True)

# df_en_annual.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\t1.xlsx")


### step 7. 연간 전기E의 Data가 존재하는 매칭표제부PK만 도출 #################################################################

df_result = pd.DataFrame()
for t in df_en_annual['에너지종류'].unique():
    df_ = df_en_annual[df_en_annual['에너지종류']==t]
    if t == '전기':
        df_ = df_[df_['count_energy']%12==0]    ## 계량기가 두개이상일 경우 여기서 각각 개별적으로 연간데이터가 모두 존재할 때는 12로 나누어져야 하기 때문에 나머지가 0이 되어야 함.
    else:
        df_ = df_   # 전기를 제외한 나머지에 대해서는 굳이 연간 데이터 모두 존재하지 않아도 넘어감.
    df_result = pd.concat([df_result,df_])

print('# of PK5 : ', df_result['매칭표제부PK'].nunique())       # of PK5 : 2950
print('# of data5 : ',len(df_result))                          # of data5 :  21112
print(' ')

# 매칭표제부별로 전기 data 갯수가 0개인 경우도 나머지가 0이므로 이를 제거해야 함.
electric_PKs = df_result[df_result['에너지종류'] == '전기']['매칭표제부PK'].unique()
df_en_annual = df_result[df_result['매칭표제부PK'].isin(electric_PKs)]

print('count(전기) = 12')
print('# of PK5 : ', df_en_annual['매칭표제부PK'].nunique())     # of PK5 :  2875
print('# of data5 : ',len(df_en_annual))                        # of data5 :  20777
print(' ')


### step 8. 연도별로 전기가 모두 존재하는 PK의 연도만 출력 ####################################################################

# 각 PK를 기준으로 그룹화
valid_dfs = []

# 각 PK를 기준으로 그룹화
grouped = df_en_annual.groupby('매칭표제부PK')

# 그룹 내에서 전기와 도시가스가 모두 있는 연도 찾기
for _, group in grouped:
    # 전기가 존재하는 연도 찾기
    electricity_years = group[group['에너지종류'] == '전기']['year_use'].unique()
    is_valid_year = group['year_use'].isin(electricity_years)
    valid_dfs.append(group[is_valid_year])

# 유효한 연도에 해당하는 행만 포함하는 최종 데이터 프레임 생성
df_en_annual1 = pd.concat(valid_dfs)


print('# of PK6 : ', df_en_annual1['매칭표제부PK'].nunique())      ## of PK5 :  1552
print('# of data6 : ',len(df_en_annual1))                        ## of data5 :  12422
print(' ')

# df_en_annual1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\sample1.xlsx")

### step 9. 데이터프레임 정리/연간총에너지 산정 ##############################################################################
df_en_annual3 = df_en_annual1[['매칭표제부PK', '에너지종류', 'year_use', 'USE_QTY_kWh','electricity_kWh','gas_kWh','districtheat_kWh']]

df_en_annual1 = df_en_annual3.groupby(['매칭표제부PK', 'year_use']).agg({
    'USE_QTY_kWh': 'sum',
    'electricity_kWh': 'sum',
    'gas_kWh': 'sum',
    'districtheat_kWh': 'sum'
}).reset_index()

print('df_en_annual : 최종_연간 총 에너지사용량')
print("# of PK7 : ",df_en_annual1['매칭표제부PK'].nunique())      # of PK2 :  2875
print('# of data7 : ', df_en_annual1.shape[0])                  # of data2 :  20777
print(' ')
# df_en_annual1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\에너지사용량(편집).xlsx")



##################################### 건축물대장_층별개요  ##########################################################################################################
print("데이터셋3. 건축물대장_층별개요 주용도(의료시설)비율 산정 ################################################################################")
print(' ')


### step 1. 층별개요 데이터 input ########################################################################################

## 층별개요 반영
file_path = os.path.join(base_dir, filenames_bldg)
df_bldg2 = pd.read_excel(file_path, sheet_name='층별개요')

print('BEFORE : 층별개요_건축물대장')
print("# of PK1 : ", df_bldg2['매칭표제부PK'].nunique())  # of PK1 :  7581
print('# of data1 : ', df_bldg2.shape[0])               # of data1 :  57007
print(' ')

### 참고용. 주용도코드명 체크  ############################################################################################

print('참고용: 주용도코드명 체크')
print("# of 주용도코드명 : ", df_bldg2['주용도코드명'].nunique())  # of 주용도코드명 :  105   / 78   / 109   / 88    / 105   / 104   / 63    (주용도코드명=nan/면적(㎡)=0인 것 포함: 추후 영향을 주기 때문)
print('# of data : ', df_bldg2.shape[0])                       # of data :       15854 / 8915 / 17472 / 12976 / 16631 / 16612 / 4855
print(' ')
print(df_bldg2['주용도코드명'].unique().tolist())
print(' ')


# # ['병원', '기타제1종근린생활시설', '기타제2종근린생활시설', '요양병원', '기숙사', '연립주택', '노인복지시설', '단독주택', '치과병원', '의원', '소매점', '휴게음식점', '주차장', '장례식장',
# # '종합병원', '일반음식점', '요양소', '사무소', '의료기기판매소', '기타병원', '의료시설', '기타의료시설', '통신용시설', '기타공연장', '일반공장', '골프연습장', '학원', '한방병원', '미용원',
# # '창고', '조산원', '산부인과병원', '조산소', '체력단련장', '부동산중개사무소', '정신병원', '제과점', '한의원', '기타사무소', '기타일반업무시설', '목욕장', '기타교육연구시설', '볼링장',
# # '직업훈련소', '금융업소', '부대시설', '제조업소', '기타공공시설', '세탁소', '이용원', '노래연습장', '기타근린생활시설', '기원', '멀티미디어문화컨텐츠설비제공업소', '유흥주점', '사회복지시설',
# # '치과의원', '회의장', '수녀원', '단란주점', '기타창고시설', '기타노유자시설', '관광호텔', '정수장', '산후조리원', '이(미)용원', '교회', '안마시술소', '교습소', '일반목욕장', '에어로빅장',
# # '안마원', '수리점', '당구장', '기타 운동시설', '연구소', '게임제공업소', '인터넷컴퓨터게임시설제공업소', '독서실', '기타발전시설', '동물병원', '기타판매시설', '자동차영업소', '오피스텔',
# # '상점', '다가구주택', '의약품판매소', '서점(1종근.생미해당)', '일반창고', '비디오물감상실', '체육도장', '방송국', '기타전시장', '여관', '교육(연수)원', '체육장', '공공시설', '일반업무시설',
# # '기타공장', '기타묘지관련시설', '어린이집', '탁구장', '음악당', '제1종근린생활시설','NaN']

### 주용도 비율 산정(고려대 주용도 혼합도와 다름) #############################################################################
df_bldg2['주용도코드명'].fillna('Unknown', inplace=True)

df_bldg2_group = df_bldg2.pivot_table(index='매칭표제부PK', columns='주용도코드명', values='면적(㎡)', aggfunc='sum')

hos_list = ['병원', '요양병원', '종합병원', '기타병원', '의료시설', '치과병원', '한방병원', '정신병원', '산부인과병원', '기타의료시설', '의원', '조산원', '한의원',
            '치과의원', '산후조리원', '조산소','전염병원','격리병원','기타격리병원','요양소']
# ※참고 어디까지를 의료시설로 봐야 할지 고민 ex)요양병원의 노인복지시설, 종합병원의 장례식장
# 앞선 데이터셋2의 종별코드명에서 6개의 병원type에 대해 암호화요양코드 1개당 매칭표제부 1개를 매칭했기 때문에 병원에 해당되는 용도인 의원까지 지정함.
existing_hos_list = [col for col in hos_list if col in df_bldg2_group.columns]
hospital_related = df_bldg2_group[existing_hos_list].sum(axis=1)

# 병원 관련 열들을 제외한 나머지 열을 합침
other = df_bldg2_group.drop(columns=existing_hos_list).sum(axis=1)

# 결과를 새로운 데이터프레임으로 만듦
result = pd.DataFrame({
    '병원 면적': hospital_related,
    '기타 면적': other
}, index=df_bldg2_group.index)

result['면적 합계'] = result.sum(axis=1)

result_ratio = pd.DataFrame()
result_ratio['주용도(의료시설) 비율(%)'] = result['병원 면적'] * 100 / result['면적 합계']
result_ratio['기타 면적 비율(%)'] = result['기타 면적'] * 100 / result['면적 합계']
result_ratio1 = pd.concat([result['면적 합계'], result_ratio], axis=1)



result_ratio=result_ratio.reset_index()



print('주용도비율산정 갯수')
print("# of PK3 : ", result_ratio['매칭표제부PK'].nunique())     # of PK3 :  2875
print('# of data3 : ', result_ratio.shape[0])                  # of data3 :  11901
print(' ')
print(' ')




result_ratio = result_ratio[result_ratio['주용도(의료시설) 비율(%)'] >= 75]
print("# of PK : ", result_ratio['매칭표제부PK'].nunique())  # of PK :    1429 / 786  / 1550 / 1055 / 1470 / 1465 / 353
print('# of data : ', result_ratio.shape[0])               # of data :  6046 / 3406 / 6547 / 4449 / 6200 / 6188 / 1448
print(' ')




### 필터_건축물대장(일반)+필터_연간총에너지사용량+ 필터_층별개요>>>>>Data Merging###########################################################

df_combined1 = pd.merge(left=df_bldg1, right= df_en_annual1, how='inner', on='매칭표제부PK')
df_combined1 = pd.merge(left=df_combined1, right=result_ratio, how='inner', on='매칭표제부PK')

print('df_combined1 : Merged data')
print("# of PK3 : ", df_combined1['매칭표제부PK'].nunique())     # of PK3 :  2875
print('# of data3 : ', df_combined1.shape[0])                  # of data3 :  11901
print(' ')
print(' ')



### 데이터셋1. 건축물대장과 에너지사용량 병합 excel Data로 export ###########################################################
df_combined1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\d.xlsx")



##################################### 데이터셋2. 건강보험심사평가원 데이터 ####################################################################################################
print('데이터셋2. 건강보험심사평가원 데이터 ################################################################################')
print(' ')


### step 1. 데이터셋2 Data input ########################################################################################

dir_path = r"C:\Lab\01_Research\01_DataNet\데이터무덤\건강보험심사평가원_전국 병의원 및 약국 현황 2022.9"

## Before : 병원정보서비스
filename = "1.병원정보서비스 2022.10..csv"
file_path = os.path.join(dir_path,filename)
hos_00 = pd.read_csv(file_path)

print('Before : 병원정보서비스')
print("# of PK1 : ", hos_00['mgm_bld_pk'].nunique())    # of PK1 :  45181
print('# of data1 : ',len(hos_00))                       # of data1 :  76032
print(' ')




## Before : 개별 데이터 선택 부분
filename = "3.의료기관별상세정보서비스_01_시설정보_2022.10..csv"
# filename = "4.의료기관별상세정보서비스_02_세부정보_2022.10..csv"
# filename = "5.의료기관별상세정보서비스_03_진료과목정보_2022.10..csv"
# filename = "7.의료기관별상세정보서비스_05_의료장비정보_2022.10..csv"
# filename = "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv"
# filename = "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv"
# filename = "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv"

file_path = os.path.join(dir_path,filename)
hos_01 = pd.read_csv(file_path, encoding='cp949')

print('Before :',filename)
print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())     # of PK1 :    52895  / 16106 / 41505  / 25087 / 6999  / 3562  / 5510
print('# of data1 : ',len(hos_01))                       # of data1 :  100335 / 20627 / 365884 / 57810 / 16206 / 12114 / 6904
print(' ')

### step 2-1. 개별 데이터 전처리 # ########################################################################################

### 01_시설정보
if filename == "3.의료기관별상세정보서비스_01_시설정보_202309.csv":
    C = ['일반입원실상급병상수', '일반입원실일반병상수', '성인중환자병상수', '소아중환자병상수',
         '신생아중환자병상수', '정신과폐쇄상급병상수', '정신과폐쇄일반병상수', '격리병실병상수',
         '무균치료실병상수', '분만실병상수', '수술실병상수', '응급실병상수', '물리치료실병상수']
    hos_01['총병상수'] = hos_01[C].sum(axis=1)

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())  # of PK1 :  52895
    print('# of data1 : ', len(hos_01))                   # of data1 :  100335
    print(' ')

### 02_세부정보
if filename == "4.의료기관별상세정보서비스_02_세부정보_202309.csv":
    hos_01.dropna(subset=hos_01.columns[16:29], how='all', inplace=True)

    def convert_and_adjust_time_columns(df, start_col, end_col):
        df[start_col] = pd.to_numeric(df[start_col], errors='coerce')
        df[end_col] = pd.to_numeric(df[end_col], errors='coerce')

        for col in [start_col, end_col]:
            df[col] = df[col].apply(lambda x: pd.to_datetime(str(int(x)).zfill(4), format='%H%M', errors='coerce') if not pd.isna(x) else x)
        return df

    day_list = ['일', '월', '화', '수', '목', '금', '토']
    for day in day_list:
        start_col = f'진료시작시간_{day}'
        end_col = f'진료종료시간_{day}'
        hos_01 = convert_and_adjust_time_columns(hos_01, start_col, end_col)


    for day in day_list:
        col1 = f'진료시간_{day}'
        col2 = f'진료시작시간_{day}'
        col3 = f'진료종료시간_{day}'
        hos_01[col1] = abs((hos_01[col3] - hos_01[col2]).dt.total_seconds() / 60)
        hos_01[col1].replace([np.inf, -np.inf], np.nan, inplace=True)
        hos_01[col1] = hos_01[col1].apply(lambda x: int(x) if pd.notna(x) else x)

    hos_01['진료시간'] = hos_01[[f'진료시간_{day}' for day in day_list]].sum(axis=1)
    hos_01 = hos_01[hos_01['진료시간'] > 0]

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())    # of PK1 :  14312
    print('# of data1 : ', len(hos_01))                     # of data1 :  18054
    print(' ')

### 03_진료과목정보
if filename == "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv":
    # NaN 값을 빈 문자열로 대체
    hos_01 = hos_01.fillna('')
    subject_counts = hos_01.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='총 진료과목코드 수')

    hos_01_pivot  = hos_01.pivot_table(
        values=['과목별 전문의수'],   #'선택진료 의사수'는 없기 때문에 제거
        index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
        columns=['진료과목코드명']
    )

    hos_01_pivot.columns = [f'{col[1]}_{col[0]}' for col in hos_01_pivot.columns]
    hos_01_pivot.reset_index(inplace=True)
    hos_01_pivot['총 전문의수'] = hos_01_pivot.filter(like='_과목별 전문의수').sum(axis=1)
    hos_01 = pd.merge(hos_01_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='inner')
    # hos_01 = hos_01.replace(0, np.nan)
    hos_01 = hos_01.replace('', np.nan)

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())  # of PK1 :    41505
    print('# of data1 : ', len(hos_01))                   # of data1 :  72981
    print(' ')

### 05_의료장비정보
if filename == "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv":
    hos_01 = hos_01.fillna('')
    hos_01 = hos_01.pivot_table(
        index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], values="장비대수", columns='장비코드명').reset_index()
    hos_01 = hos_01.replace('', np.nan)
    hos_01 = hos_01.drop(columns=['怨⑤寃ш린'])
    columns_to_sum = ['CT', 'MRI', '골밀도검사기', '양전자단층촬영기 (PET)', '유방촬영장치',
                      '종양치료기 (Cyber Knife)', '종양치료기 (Gamma Knife)', '종양치료기 (양성자치료기)',
                      '체외충격파쇄석기', '초음파영상진단기', '콘빔CT', '혈액투석을위한인공신장기']
    # hos_01[columns_to_sum] = hos_01[columns_to_sum].replace(np.nan, 0)
    hos_01['장비수'] = hos_01[columns_to_sum].sum(axis=1)

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())    # of PK1 :    25087
    print('# of data1 : ', len(hos_01))                     # of data1 :  37627
    print(' ')

### 06_식대가산정보
if filename == "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv":
    hos_01 = hos_01.fillna('')
    # 피벗 테이블 생성
    hos_01 = hos_01.pivot_table(
        index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
        columns='유형코드명',
        values=['산정인원수', '일반식 가산여부', '치료식 등급'],
        aggfunc = 'first'
    )
    hos_01.columns = [f'{col[1]}_{col[0]}' for col in hos_01.columns]
    hos_01.reset_index(inplace=True)
    hos_01 = hos_01.replace('', np.nan)
    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())    # of PK1 :    6999
    print('# of data1 : ', len(hos_01))                     # of data1 :  8103
    print(' ')

### 07_간호등급정보
if filename == "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv":
    # NaN 값을 빈 문자열로 대체
    hos_01 = hos_01.fillna('')
    subject_counts = hos_01.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='유형코드명_개수')
    hos_01_pivot  = pd.pivot_table(
        hos_01,
        values=['간호등급'],
        index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
        columns=['유형코드명']
    )
    # # Flatten the MultiIndex columns
    hos_01_pivot.columns = [f'{col[1]}_{col[0]}' for col in hos_01_pivot.columns]
    hos_01_pivot.reset_index(inplace=True)
    hos_01_pivot['합계_간호등급'] = hos_01_pivot.filter(like='_간호등급').sum(axis=1)
    hos_01 = pd.merge(hos_01_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='inner')
    hos_01 = hos_01.replace('', np.nan)

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())    # of PK1 :    3562
    print('# of data1 : ', len(hos_01))                     # of data1 :  3967
    print(' ')

### 08_특수진료정보
if filename == "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv":
    # NaN 값을 빈 문자열로 대체
    hos_01 = hos_01.fillna('')
    subject_counts = hos_01.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='유형코드명_개수')
    hos_01_pivot = hos_01.pivot_table(index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], columns='검색코드명', aggfunc='size', fill_value=0)

    hos_01 = pd.merge(hos_01_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='inner')
    hos_01 = hos_01.replace('', np.nan)

    print('Before : 필터')
    print("# of PK1 : ", hos_01['mgm_bld_pk'].nunique())  # of PK1 :    5510
    print('# of data1 : ', len(hos_01))                   # of data1 :  6238
    print(' ')



### step 2-2. 통합 데이터 전처리 ###############################################################################################

def Data_Programming(A):
    # mgm_bld_pk ≠ NaN & mgm_upper_bld_pk = NaN
    hos = A[A['mgm_bld_pk'].notna()]    ## 표제부!=NaN
    hos = hos[hos['mgm_upper_bld_pk'].isna()]       ## 총괄표제부==NaN

    hos['mgm_bld_pk'] = hos['mgm_bld_pk'].str.split(',')
    hos = hos.explode('mgm_bld_pk')

    # 하나의 암호화요양기호에 하나의 표제부PK만 추출
    hos['PK당요양기호'] = hos.groupby(['mgm_bld_pk'])['암호화요양기호'].transform('count')      ## 매칭표제부 당 암호화요양기호 갯수 산정
    hos['요양기호당PK'] = hos.groupby(['암호화요양기호'])['mgm_bld_pk'].transform('count')      ## 암호화요양기호 당 매칭표제부 갯수 산정
    hos = hos[hos['PK당요양기호'] == 1].copy()       ## 표제부당 암호화요양기호가 1개인 것만 추출
    hos = hos[hos['요양기호당PK'] == 1].copy()       ## 암호화요양기호당 표제부가 1개인 것만 추출



    # ## 전처리된 건축물대장 PK list입력
    # PK_list = df_bldg1['매칭표제부PK'].unique()
    # hos = hos[hos['mgm_bld_pk'].isin(PK_list)]

    hos.drop(columns={'PK당요양기호','요양기호당PK'}, inplace=True)     ## PK당요양기호, 요양기호당PK 열 제거
    return hos


## After : 병원정보서비스_Data
hos_00=Data_Programming(hos_00)

print('After : 병원정보서비스')
print("# of PK2 : ", hos_00['mgm_bld_pk'].nunique())     # of PK2 :  1791
print('# of data2 : ',len(hos_00))                       # of data2 :  1791

## After : 개별_Data
hos_01=Data_Programming(hos_01)

print('After : ',filename)
print("# of PK2 : ", hos_01['mgm_bld_pk'].nunique())     # of PK2 :    1661 / 959 / 1776 / 1277 / 1724 / 1696 / 440
print('# of data2 : ',len(hos_01))                       # of data2 :  1661 / 959 / 1776 / 1277 / 1724 / 1696 / 440
print(' ')



### step 3. 데이터셋2. 병원정보서비스 + 시설정보 병합 ########################################################################
if filename == "3.의료기관별상세정보서비스_01_시설정보_202309.csv":
    hos_merge = pd.merge(left=hos_00, right=hos_01, how='inner', on=['암호화요양기호','mgm_bld_pk','종별코드명']) ## 암호화요양기호 및 매칭표제부 3개 기준 병합
else:
    hos_merge = pd.merge(left=hos_00, right=hos_01, how='inner', on=['암호화요양기호','mgm_bld_pk']) ## 암호화요양기호 및 매칭표제부 2개 기준 병합

print('After : Merged data')
print("# of PK3 : ", hos_merge['mgm_bld_pk'].nunique())     # of PK3 :    1636 / 854 / 1762 / 1189 / 1584 / 1550 / 383
print('# of data3 : ',len(hos_merge))                       # of data3 :  1636 / 854 / 1762 / 1189 / 1584 / 1550 / 383
print(' ')



### step 4. 데이터셋2. 종별코드명 = '병원', '치과병원','한방병원','요양병원','정신병원','종합병원' ################################
hos_merge = hos_merge[hos_merge['종별코드명'].isin(['병원', '치과병원','한방병원','요양병원','정신병원','종합병원'])] ##대상 병원 필터
# hos_merge = hos_merge[hos_merge['종별코드명'].isin(['병원', '치과병원','한방병원','요양병원','정신병원','종합병원','상급종합'])] ##대상 병원 필터

print('병원, 치과병원, 한방병원, 요양병원, 정신병원, 종합병원')
print("# of PK4 : ", hos_merge['mgm_bld_pk'].nunique())     # of PK4 :    1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print('# of data4 : ',len(hos_merge))                       # of data4 :  1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print(' ')



hos_merge1 = hos_merge[hos_merge['종별코드명'].isin(['치과병원'])] ##대상 병원 필터

hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\치과병원만.xlsx")

hos_merge=hos_merge[hos_merge['총의사수']>0]
print('의사수>0')
print("# of PK4 : ", hos_merge['mgm_bld_pk'].nunique())     # of PK4 :    1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print('# of data4 : ',len(hos_merge))                       # of data4 :  1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print(' ')


hos_merge = hos_merge[
    (
        (hos_merge['종별코드명'].isin(['종합병원']) & (hos_merge['총병상수'] >= 100)) |
        (hos_merge['종별코드명'].isin(['병원', '한방병원']) & (hos_merge['총병상수'] >= 30)) |
        (hos_merge['종별코드명'].isin(['정신병원', '치과병원', '요양병원']))
    )
]

print('병원종별 병상수 필터')

print("# of PK4 : ", hos_merge['mgm_bld_pk'].nunique())     # of PK4 :    1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print('# of data4 : ',len(hos_merge))                       # of data4 :  1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print(' ')


hos_merge1 = hos_merge[hos_merge['종별코드명'].isin(['치과병원'])] ##대상 병원 필터
print("# of PK4 : ", hos_merge1['mgm_bld_pk'].nunique())     # of PK4 :    1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print('# of data4 : ',len(hos_merge1))                       # of data4 :  1512 / 824 / 1635 / 1111 / 1548 / 1546 / 374
print(' ')


### 데이터셋2. 병원정보서비스 + 시설정보 병합 excel Data로 export ############################################################
hos_merge.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_데이터셋2..xlsx")


##################################### 데이터셋1 + 데이터셋2 병합 ##########################################################################################################
print("데이터셋1 + 데이터셋2 병합 #########################################################################################")



### 데이터셋1 + 데이터셋2 병합 ###########################################################################################
hos_merge1 = pd.merge(left=df_combined1, right=hos_merge, how='inner',left_on=['매칭표제부PK'], right_on=['mgm_bld_pk'])

print("# of PK5 : ", hos_merge1['매칭표제부PK'].nunique())       # of PK5 :    1429 /  786 / 1550 / 1055 / 1470 / 1465 / 353
print('# of data5 : ',len(hos_merge1))                         # of data5 :  6046 / 3406 / 6547 / 4449 / 6200 / 6188 / 1448
print(' ')
print(' ')




if filename == "3.의료기관별상세정보서비스_01_시설정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_01_시설정보.xlsx")
if filename == "4.의료기관별상세정보서비스_02_세부정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_02_세부정보.xlsx")
if filename == "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv":
    hos_merge1.to_excel(r"CC:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_03_진료과목정보.xlsx")
if filename == "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_05_의료장비정보.xlsx")
if filename == "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_06_식대가산정보.xlsx")
if filename == "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_07_간호등급정보.xlsx")
if filename == "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv":
    hos_merge1.to_excel(r"C:\Lab\01_Research\01_DataNet\01_python file\전처리데이터\데이터넷_1_08_특수진료정보.xlsx")



# ### 데이터셋1, 2, 3 병합 연도별로 excel Data로 export ########################################################################
# print('데이터셋1, 2, 3 병합 연도별 구분############################')
# print(' ')
#
# def count_dataset(Da,A):
#     ho=Da[Da['year_use'] == A]
#
#     print(A)
#     print("# of PK : ", ho['매칭표제부PK'].nunique())
#     print('# of data : ',len(ho))
#     print(' ')
#     return ho
#
# #총 면적과 연면적 비교
# if filename == "3.의료기관별상세정보서비스_01_시설정보_202309.csv":
#     q2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_01_시설정보.xlsx")# of PK :  1139
#     q2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_01_시설정보.xlsx")# of PK :  1211
#     q2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_01_시설정보.xlsx")# of PK :  1206
#     q2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_01_시설정보.xlsx")# of PK :  1209
#     q2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_01_시설정보.xlsx")# of PK :  1281
#
# if filename == "4.의료기관별상세정보서비스_02_세부정보_202309.csv":
#     w2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_02_세부정보.xlsx")# of PK :  659
#     w2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_02_세부정보.xlsx")# of PK :  691
#     w2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_02_세부정보.xlsx")# of PK :  677
#     w2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_02_세부정보.xlsx")# of PK :  669
#     w2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_02_세부정보.xlsx")# of PK :  710
#
# if filename == "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv":
#     e2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_03_진료과목정보.xlsx")# of PK :  1234
#     e2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_03_진료과목정보.xlsx")# of PK :  1311
#     e2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_03_진료과목정보.xlsx")# of PK :  1301
#     e2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_03_진료과목정보.xlsx")# of PK :  1311
#     e2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_03_진료과목정보.xlsx")# of PK :  1390
#
# if filename == "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv":
#     r2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_05_의료장비정보.xlsx")# of PK :  846
#     r2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_05_의료장비정보.xlsx")# of PK :  887
#     r2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_05_의료장비정보.xlsx")# of PK :  877
#     r2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_05_의료장비정보.xlsx")# of PK :  893
#     r2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_05_의료장비정보.xlsx")# of PK :  946
#
# if filename == "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv":
#     t2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_06_식대가산정보.xlsx")# of PK :  1166
#     t2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_06_식대가산정보.xlsx")# of PK :  1240
#     t2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_06_식대가산정보.xlsx")# of PK :  1230
#     t2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_06_식대가산정보.xlsx")# of PK :  1240
#     t2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_06_식대가산정보.xlsx")# of PK :  1324
#
# if filename == "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv":
#     y2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_07_간호등급정보.xlsx")# of PK :  1162
#     y2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_07_간호등급정보.xlsx")# of PK :  1239
#     y2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_07_간호등급정보.xlsx")# of PK :  1233
#     y2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_07_간호등급정보.xlsx")# of PK :  1237
#     y2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_07_간호등급정보.xlsx")# of PK :  1317
#
# if filename == "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv":
#     u2018=count_dataset(hos_merge1, 2018).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_08_특수진료정보.xlsx")# of PK :  264
#     u2019=count_dataset(hos_merge1, 2019).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_08_특수진료정보.xlsx")# of PK :  290
#     u2020=count_dataset(hos_merge1, 2020).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_08_특수진료정보.xlsx")# of PK :  283
#     u2021=count_dataset(hos_merge1, 2021).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_08_특수진료정보.xlsx")# of PK :  294
#     u2022=count_dataset(hos_merge1, 2022).to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_08_특수진료정보.xlsx")# of PK :  317


# ### 데이터셋1, 2, 3 병합 연도별 전체 병합 excel Data로 export ###################################################################################################################################
# # 이건 위의 코드를 개별적으로 다 하고 진행할 것
# # 독립변수별 추출 시 파일 투입 및 병합
#
# def input_data_year(A,B):
#     base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"
#
#     # Construct the directory path
#     dir_path = os.path.join(base_dir, str(A))
#
#     # Construct the filename
#     filename1 = "데이터넷_1_" + str(A) + "_"+B+".xlsx"
#
#     # Construct the full file path
#     file_path1 = os.path.join(dir_path, filename1)
#     # Read the Excel file into a DataFrame
#
#     export_data = pd.read_excel(file_path1)
#     return export_data
# year_input=2018
# df_01=input_data_year(year_input,"01_시설정보")
# df_02=input_data_year(year_input,"02_세부정보")
# df_03=input_data_year(year_input,"03_진료과목정보")
# df_05=input_data_year(year_input,"05_의료장비정보")
# df_06=input_data_year(year_input,"06_식대가산정보")
# df_07=input_data_year(year_input,"07_간호등급정보")
# df_08=input_data_year(year_input,"08_특수진료정보")
#
#
# ### 최종 병합본 연도별 교집합 excel Data로 export ########################################################################
# # 유효한 데이터프레임만 선택
# dataframes = [df for df in [df_01, df_02, df_03, df_05, df_06, df_07] if df is not None]
# # dataframes = [df for df in [df_01, df_02, df_03, df_05, df_06, df_07, df_08] if df is not None]
#
# # 첫 번째 데이터프레임을 기준으로 나머지 데이터프레임을 순차적으로 병합
# merged_df = dataframes[0]
# for df in dataframes[1:]:
#     merged_df = pd.merge(merged_df, df, on=['암호화요양기호', 'mgm_bld_pk', '종별코드명'], how='inner')
#
#
# print(merged_df.head())
# print("# of PK : ", merged_df['mgm_bld_pk'].nunique())  # of PK :   [2018: 399(08_특수진료정보 제외) / 125], [2019: 424(08_특수진료정보 제외) / 141], [2020: 412(08_특수진료정보 제외) / 135], [2021: 402(08_특수진료정보 제외) / 138], [2022: 442(08_특수진료정보 제외) / 155]
# print('# of data : ', len(merged_df))                   # of data : [2018: 399(08_특수진료정보 제외) / 125], [2019: 424(08_특수진료정보 제외) / 141], [2020: 412(08_특수진료정보 제외) / 135], [2021: 402(08_특수진료정보 제외) / 138], [2022: 442(08_특수진료정보 제외) / 155]
# print(' ')
