import os
import numpy as np
import pandas as pd

##################################### 데이터셋1. 건축물대장 + 에너지사용량 ####################################################################################################

### 데이터셋1. Data input #####################################################################################################################################################

## 건축물대장과 에너지사용량 동일 경로 입력
base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

## 건축물대장 excel 입력
filenames_bldg="데이터넷_의료시설(건축물대장).xlsx"
sheetnames_bldg="건축물대장(표제부)"


### 에너지사용량 excel 입력
filenames_energy = ["데이터넷_의료시설(에너지사용량_2018년).xlsx",
                    "데이터넷_의료시설(에너지사용량_2019년).xlsx",
                    "데이터넷_의료시설(에너지사용량_2020년).xlsx",
                    "데이터넷_의료시설(에너지사용량_2021년).xlsx",
                    "데이터넷_의료시설(에너지사용량_2022년).xlsx"]
sheetnames_energy = ['표제부_에너지사용량_계량기_2018년',
                     '표제부_에너지사용량_계량기_2019년',
                     '표제부_에너지사용량_계량기_2020년',
                     '표제부_에너지사용량_계량기_2021년',
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
df_bldg1 = df_bldg1[(df_bldg1['용적률산정연면적(㎡)']>0)&(df_bldg1['연면적(㎡)']>0)]

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

PK_list = df_bldg1['매칭표제부PK'].unique()              ## 전처리된 건축물대장 PK list입력
df_en = df_en[df_en['매칭표제부PK'].isin(PK_list)]

print('전처리된 건축물대장 PK list로 필터')
print('# of PK2 : ', df_en['매칭표제부PK'].nunique())    ## of PK2 : 3023
print('# of data2 : ', df_en.shape[0])                 ## of data2 : 742508
print(' ')



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

df_en['USE_QTY_kWh'] = df_en.apply(lambda row: energy_conversion(row), axis=1)



### step 6. 에너지 종류에 따른 열 구분 & 연간총에너지사용량 및 갯수 산정 #####################################################

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



### step 7. 연간 전기E의 Data가 존재하는 매칭표제부PK만 도출 ###############################################################

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


### step 8. 연도별로 전기가 모두 존재하는 PK의 연도만 출력 ###############################################################

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

df_en_annual1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\sample1.xlsx")

## 데이터프레임 정리/연간총에너지 산정 #######################################################################################
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


##################################### 건축물대장 + 에너지사용량 병합 ##########################################################################################################
print('데이터셋1. 건축물대장 + 에너지사용량 병합 ###########################################################################')
print(' ')

### 필터_건축물대장(일반)#################################################################################################

print('df_bldg1 : 건축물대장')
print("# of PK1 : ", df_bldg1['매칭표제부PK'].nunique())         # of PK1 :  3758
print('# of data1 : ', df_bldg1.shape[0])                      # of data1 :  3758
print(' ')


### 필터_연간 총 에너지사용량##############################################################################################

print('df_en_annual : 연간 총 에너지사용량')
print("# of PK2 : ",df_en_annual1['매칭표제부PK'].nunique())      # of PK2 :  2875
print('# of data2 : ', df_en_annual1.shape[0])                  # of data2 :  20777
print(' ')



### 필터_건축물대장(일반)+필터_연간총에너지사용량>>>>>Data Merging###########################################################

df_combined1 = pd.merge(left=df_en_annual1, right=df_bldg1, how='left', on='매칭표제부PK')

print('df_combined1 : Merged data')
print("# of PK3 : ", df_combined1['매칭표제부PK'].nunique())     # of PK3 :  2875
print('# of data3 : ', df_combined1.shape[0])                  # of data3 :  20777
print(' ')
print(' ')

### 데이터셋1. 건축물대장과 에너지사용량 병합 excel Data로 export ###########################################################
df_combined1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_데이터셋1.xlsx")






##################################### 데이터셋2. 건강보험심사평가원 데이터 ####################################################################################################
print('데이터셋2. 건강보험심사평가원 데이터 ################################################################################')


### step 1. 데이터셋2 Data input ########################################################################################

## Before : 시설정보
filenames = [
    "1.병원정보서비스 2022.10..csv",
    "3.의료기관별상세정보서비스_01_시설정보_202309.csv",
    "4.의료기관별상세정보서비스_02_세부정보_202309.csv",
    "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv",
    "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv",
    "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv",
    "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv",
    "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv"
]

def Data_Programming_Detail(filename):
    dir_path = r"C:\Users\user\Desktop\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결"

    file_path = os.path.join(dir_path, filename)
    try:
        hos = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        hos = pd.read_csv(file_path, encoding='cp949')
    print('Before : ', filename)
    print("# of PK1 : ", hos['mgm_bld_pk'].nunique())  # of PK3 :  2875
    print('# of data1 : ', hos.shape[0])


    ### step 1. 데이터 전처리 개별 ###########################################################################################

    if filename == "5.의료기관별상세정보서비스_03_진료과목정보_202309.csv":
        # NaN 값을 빈 문자열로 대체
        hos = hos.fillna('')
        subject_counts = hos.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='총 진료과목코드 수')

        hos_pivot  = hos.pivot_table(
            values=['과목별 전문의수'],   #'선택진료 의사수'는 없기 때문에 제거
            index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
            columns=['진료과목코드명']
        )

        hos_pivot.columns = [f'{col[1]}_{col[0]}' for col in hos_pivot.columns]
        hos_pivot.reset_index(inplace=True)
        hos_pivot['총 전문의수'] = hos_pivot.filter(like='_과목별 전문의수').sum(axis=1)
        hos = pd.merge(hos_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='left')
        hos = hos.replace(0, np.nan)
        hos = hos.replace('', np.nan)


    if filename == "7.의료기관별상세정보서비스_05_의료장비정보_202309.csv":
        hos = hos.fillna('')
        # 피벗 테이블 생성
        hos = hos.pivot_table(
            index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], values="장비대수", columns='장비코드명').reset_index()
        hos = hos.replace('', np.nan)



    if filename == "8.의료기관별상세정보서비스_06_식대가산정보_202309.csv":
        hos = hos.fillna('')
        # 피벗 테이블 생성
        hos = hos.pivot_table(
            index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
            columns='유형코드명',
            values=['산정인원수', '일반식 가산여부', '치료식 등급'],
            aggfunc = 'first'
        )
        hos.columns = [f'{col[1]}_{col[0]}' for col in hos.columns]
        hos.reset_index(inplace=True)
        hos = hos.replace('', np.nan)

    if filename == "9.의료기관별상세정보서비스_07_간호등급정보_202309.csv":
        # NaN 값을 빈 문자열로 대체
        hos = hos.fillna('')
        subject_counts = hos.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='유형코드명_개수')
        hos_pivot  = pd.pivot_table(
            hos,
            values=['간호등급'],
            index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'],
            columns=['유형코드명']
        )
        # # Flatten the MultiIndex columns
        hos_pivot.columns = [f'{col[1]}_{col[0]}' for col in hos_pivot.columns]
        hos_pivot.reset_index(inplace=True)
        hos_pivot['합계_간호등급'] = hos_pivot.filter(like='_간호등급').sum(axis=1)
        hos = pd.merge(hos_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='left')
        hos = hos.replace('', np.nan)

    if filename == "10.의료기관별상세정보서비스_08_특수진료정보_202309.csv":
        # NaN 값을 빈 문자열로 대체
        hos = hos.fillna('')
        subject_counts = hos.groupby(['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk']).size().reset_index(name='유형코드명_개수')
        hos_pivot = hos.pivot_table(index=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], columns='검색코드명', aggfunc='size', fill_value=0)

        hos = pd.merge(hos_pivot, subject_counts, on=['암호화요양기호', '요양기관명', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='left')
        hos = hos.replace('', np.nan)


    print('After_Detail : ', filename)
    print("# of PK2 : ", hos['mgm_bld_pk'].nunique())
    print('# of data2 : ', hos.shape[0])
    print(' ')


    # mgm_bld_pk ≠ NaN & mgm_upper_bld_pk = NaN
    hos = hos[hos['mgm_bld_pk'].notna()]  ## 표제부!=NaN
    hos = hos[hos['mgm_upper_bld_pk'].isna()]  ## 총괄표제부==NaN

    hos['mgm_bld_pk'] = hos['mgm_bld_pk'].str.split(',')
    hos = hos.explode('mgm_bld_pk')

    # 하나의 암호화요양기호에 하나의 표제부PK만 추출
    hos['PK당요양기호'] = hos.groupby(['mgm_bld_pk'])['암호화요양기호'].transform('count')  ## 매칭표제부 당 암호화요양기호 갯수 산정
    hos['요양기호당PK'] = hos.groupby(['암호화요양기호'])['mgm_bld_pk'].transform('count')  ## 암호화요양기호 당 매칭표제부 갯수 산정
    hos = hos[hos['PK당요양기호'] == 1].copy()  ## 표제부당 암호화요양기호가 1개인 것만 추출
    hos = hos[hos['요양기호당PK'] == 1].copy()  ## 암호화요양기호당 표제부가 1개인 것만 추출

    ## 전처리된 건축물대장 PK list입력
    PK_list = df_bldg1['매칭표제부PK'].unique()
    hos = hos[hos['mgm_bld_pk'].isin(PK_list)]

    hos.drop(columns={'PK당요양기호', '요양기호당PK'}, inplace=True)  ## PK당요양기호, 요양기호당PK 열 제거

    print('After_Total : ', filename)
    print("# of PK3 : ", hos['mgm_bld_pk'].nunique())
    print('# of data3 : ', len(hos))
    print(' ')

    return hos

hos_00=Data_Programming_Detail(filenames[0])

hos_01=Data_Programming_Detail(filenames[1])

hos_02=Data_Programming_Detail(filenames[2])

hos_03=Data_Programming_Detail(filenames[3])

hos_04=Data_Programming_Detail(filenames[4])

hos_05=Data_Programming_Detail(filenames[5])

hos_06=Data_Programming_Detail(filenames[6])

# hos_07=Data_Programming_Detail(filenames[7])


### step 3. 데이터셋2. 병원정보서비스 + 시설정보 병합 ########################################################################
# 유효한 데이터프레임만 선택
dataframes = [df for df in [hos_00, hos_01, hos_02, hos_03, hos_04, hos_05, hos_06] if df is not None]
# dataframes = [df for df in [hos_00, hos_01, hos_02, hos_03, hos_04, hos_05, hos_06, hos_07] if df is not None]

# hos_00과 hos_01을 병합할 때 '종별코드명'도 병합
merged_df = pd.merge(hos_00, hos_01, on=['암호화요양기호', 'mgm_bld_pk', 'mgm_upper_bld_pk', '종별코드명'], how='inner')

# 나머지 데이터프레임을 순차적으로 병합 (inner join)
for df in dataframes[2:]:
    merged_df = pd.merge(merged_df, df, on=['암호화요양기호', 'mgm_bld_pk', 'mgm_upper_bld_pk'], how='inner')

print(merged_df.head())
print("# of PK4 : ", merged_df['mgm_bld_pk'].nunique())
print('# of data4 : ', len(merged_df))
print(' ')


### step 4. 데이터셋2. 종별코드명 = '병원', '치과병원','한방병원','요양병원','정신병원','종합병원' ###############################
hos_merge = merged_df[merged_df['종별코드명'].isin(['병원', '치과병원','한방병원','요양병원','정신병원','종합병원'])] ##대상 병원 필터

print('병원, 치과병원, 한방병원, 요양병원, 정신병원, 종합병원')
print("# of PK4 : ", hos_merge['mgm_bld_pk'].nunique())     # of PK4 :  994
print('# of data4 : ',len(hos_merge))                       # of data4 :  994
print(' ')

### 데이터셋2. 병원정보서비스 + 시설정보 병합 excel Data로 export ############################################################
hos_merge.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_데이터셋2..xlsx")



##################################### 데이터셋1 + 데이터셋2 병합 ##########################################################################################################
print("데이터셋1 + 데이터셋2 병합 #########################################################################################")



### 데이터셋1 + 데이터셋2 병합 ###########################################################################################
hos_merge1 = pd.merge(left=df_combined1, right=hos_merge, how='inner',left_on=['매칭표제부PK'], right_on=['mgm_bld_pk'])

print("# of PK5 : ", hos_merge1['매칭표제부PK'].nunique())       # of PK :  994
print('# of data5 : ',len(hos_merge1))                          # of data :  4081
print(' ')



### 데이터셋1 + 데이터셋2 병합 excel Data로 export ########################################################################

def count_dataset(Da,A):
    ho=Da[Da['year_use'] == A]

    print(A)
    print("# of PK : ", ho['매칭표제부PK'].nunique())       # of PK :  994
    print('# of data : ',len(ho))                          # of data :  4081
    print(' ')
    return ho


count_dataset(hos_merge1, 2018)
count_dataset(hos_merge1, 2019)
count_dataset(hos_merge1, 2020)
count_dataset(hos_merge1, 2021)
count_dataset(hos_merge1, 2022)