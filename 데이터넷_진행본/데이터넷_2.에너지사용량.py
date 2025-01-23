import os

import pandas as pd

def input_data_year(A,B):
    base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

    # Construct the directory path
    dir_path = os.path.join(base_dir, str(A))

    # Construct the filename
    filename1 = "데이터넷_1_" + str(A) + "_"+B+".xlsx"

    # Construct the full file path
    file_path1 = os.path.join(dir_path, filename1)
    # Read the Excel file into a DataFrame

    export_data = pd.read_excel(file_path1)
    return export_data
year_input=2018
df_01=input_data_year(year_input,"01_시설정보")
df_02=input_data_year(year_input,"02_세부정보")
df_03=input_data_year(year_input,"03_진료과목정보")
df_05=input_data_year(year_input,"05_의료장비정보")
df_06=input_data_year(year_input,"06_식대가산정보")
df_07=input_data_year(year_input,"07_간호등급정보")
df_08=input_data_year(year_input,"08_특수진료정보")


### 최종 병합본 연도별 교집합 excel Data로 export ########################################################################
# 유효한 데이터프레임만 선택
dataframes = [df for df in [df_01, df_02, df_03, df_05, df_06, df_07] if df is not None]
# dataframes = [df for df in [df_01, df_02, df_03, df_05, df_06, df_07, df_08] if df is not None]

# 첫 번째 데이터프레임을 기준으로 나머지 데이터프레임을 순차적으로 병합
merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on=['암호화요양기호', 'mgm_bld_pk', '종별코드명'], how='inner')


print(merged_df.head())
print("# of PK : ", merged_df['mgm_bld_pk'].nunique())  # of PK :   [2018: 399(08_특수진료정보 제외) / 125], [2019: 424(08_특수진료정보 제외) / 141], [2020: 412(08_특수진료정보 제외) / 135], [2021: 402(08_특수진료정보 제외) / 138], [2022: 442(08_특수진료정보 제외) / 155]
print('# of data : ', len(merged_df))                   # of data : [2018: 399(08_특수진료정보 제외) / 125], [2019: 424(08_특수진료정보 제외) / 141], [2020: 412(08_특수진료정보 제외) / 135], [2021: 402(08_특수진료정보 제외) / 138], [2022: 442(08_특수진료정보 제외) / 155]
print(' ')