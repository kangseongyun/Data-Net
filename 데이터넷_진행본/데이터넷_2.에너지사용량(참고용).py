import os
import numpy as np
import pandas as pd

def data_processing(A):
    ## Data input#######################################################################################################
    ## 건축물대장과 에너지사용량 동일 경로 입력
    base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

    ## 데이터넷_1.건축물대장.py에서 전처리된 건축물대장 excel 입력
    filenames_bldg="데이터넷_건축물대장(일반).xlsx"

    ## 에너지사용량 excel 입력
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


    ### 2018-2022의 5개년을 분석하고 싶은 경우 A에 '병합'을 입력 >> 이걸로 진행
    if A in '병합':
        df_en = pd.DataFrame()
        for i in range(0, len(filenames_energy)):
            file_path = os.path.join(base_dir, filenames_energy[i])
            df_ = pd.read_excel(file_path, sheet_name=sheetnames_energy[i])
            df_en = pd.concat([df_en, df_], ignore_index=True)
        print('Year : ','병합###########################################################################################')


    ### 개별 연도별로 분석을 하고 싶을 때는 2018-2022를 A에 입력한 대로 출력할 수 있도록 하기 위함
    if A in ['2018','2019','2020','2021','2022']:
        if A in '2018':
            sheetname = sheetnames_energy[0]
            filename = filenames_energy[0]
            print('Year : ','2018#########################################################################################')

        if A in '2019':
            sheetname = sheetnames_energy[1]
            filename = filenames_energy[1]
            print('Year : ','2019#########################################################################################')

        if A in '2020':
            sheetname = sheetnames_energy[2]
            filename = filenames_energy[2]
            print('Year : ','2020#########################################################################################')

        if A in '2021':
            sheetname = sheetnames_energy[3]
            filename = filenames_energy[3]
            print('Year : ','2021#########################################################################################')

        if A in '2022':
            sheetname = sheetnames_energy[4]
            filename = filenames_energy[4]
            print('Year : ','2022#########################################################################################')
        file_path = os.path.join(base_dir, filename)
        df_en = pd.read_excel(file_path, sheet_name=sheetname, dtype={'단위코드': str})

    df_en.rename(columns={'매칭총괄표제부PK': '매칭표제부PK'}, inplace=True) ## 기존 매칭총괄표제부PK를 매칭표제부PK로 정정
    df_en['사용년월'] = pd.to_datetime(df_en['사용년월'], format='%Y%m')
    df_en['year_use'] = df_en['사용년월'].dt.year
    df_en['month_use'] = df_en['사용년월'].dt.month
    print('BEFORE')
    print('# of PK1 : ', df_en['매칭표제부PK'].nunique())    ## of PK : 병합(4048)/2018(3900)/2019(3940)/2020(3994)/2021(4057)/2022(4048)
    print('# of ROW1 : ', df_en.shape[0])                  ## of ROW1 : 병합(206785)/2018(191611)/2019(195970)/2020(201078)/2021(206262)/2022(206785)


    ## 건축물대장에 있는 매칭표제부PK만 사용 #################################################################################
    file_path_bldg = os.path.join(base_dir, filenames_bldg)     ## 전처리된 Data input
    df_bldg = pd.read_excel(file_path_bldg)
    PK_list = df_bldg['매칭표제부PK'].unique()
    df_en = df_en[df_en['매칭표제부PK'].isin(PK_list)]
    print('AFTER')
    print('# of PK2 : ', df_en['매칭표제부PK'].nunique())    ## of PK2 : 병합(2989)/2018(2902)/2019(2930)/2020(2958)/2021(2995)/2022(2989)
    print('# of ROW2 : ', df_en.shape[0])                   ## of ROW2 : 병합(152975)/2018(142681)/2019(145648)/2020(148685)/2021(152519)/2022(152975)

    ## 사용량 NaN과 >0 제거 ##############################################################################################
    df_en = df_en[~df_en['사용량'].isna()]
    df_en['사용량'] = df_en['사용량'].astype(float)
    df_en = df_en[df_en['사용량'] > 0]
    print('# of PK3 : ', df_en['매칭표제부PK'].nunique())    ## of PK3 : 병합(2967)/2018(2881)/2019(2904)/2020(3630)/2021(2977)/2022(2967)
    print('# of ROW3 : ',len(df_en))                        ## of ROW3 :  병합(142499)/2018(133027)/2019(136139)/2020(138953)/2021(142468)/2022(142499)


    ## 사용시작일자와 사용종료일자 NaN제거 #################################################################################
    # ## 해당 부분은 제거시 표제부pk가 줄어들고 밑에 코드에서 연간 총에너지 사용량 수식을 쓰면 사라지기 때문에 데이터를 최대한 확보를 위해 일단 제거 안하는 걸로 하기로 했음(안형욱 교수님 회의 후 결정 됨.)
    # df_en = df_en[(~df_en['사용시작일자'].isna()) & (~df_en['사용종료일자'].isna())]
    # df_en['사용시작일자'] = pd.to_datetime(df_en['사용시작일자'], format='%Y%m%d', errors='coerce')
    # df_en['사용종료일자'] = pd.to_datetime(df_en['사용종료일자'], format='%Y%m%d', errors='coerce')
    # print('# of PK : ', df_en['매칭표제부PK'].nunique())     ## of PK : 병합(2958)/2018(2875)/2019(2896)/2020(2931)/2021(2969)/2022(2958)
    # print('# of ROW : ',len(df_en))                         ## of PK :  병합(134645)/2018(126136)/2019(128823)/2020(131245)/2021(134648)/2022(134645)


    ### 에너지 종류 표시(전기/도시가스/지역난방) #############################################################################
    df_en = df_en[~df_en['에너지공급기관코드'].isna()] ## '에너지공급기관코드'열에 모두 value가 존재하므로 의미가 없음.
    df_en['에너지공급기관코드'] = df_en['에너지공급기관코드'].astype(str)
    filename = '데이터넷_의료시설(에너지사용량_2018년).xlsx'
    file_path = os.path.join(base_dir, filename)
    df_list = pd.read_excel(file_path, sheet_name='에너지용도코드', dtype=str)

    mapping_data1 = df_list[['기관코드', '에너지종류']]
    df_en['에너지종류'] = df_en.apply(lambda row: mapping_data1[mapping_data1['기관코드'] == row['에너지공급기관코드']]['에너지종류'].values[0] if row['에너지공급기관코드'] in mapping_data1['기관코드'].values else '', axis=1)
    # 에너지용도코드의 기관코드를 통해 에너지 종류(전기/도시가스/지역난방) 매칭

    ## 단위코드 열을 kWh기준 단위 환산 #################################################################################################
    df_en = df_en[~df_en['단위코드'].isna()] ## '단위코드'열에 모두 value가 존재하므로 의미가 없음.
    def energy_conversion(row):
        if row['단위코드'] == '01': #
            x = 1
        elif row['단위코드'] == '02': #
            x = 42.7 * 1 / 3.6
        elif row['단위코드'] == '03': #
            x = 1 / 0.860 * 1000
        elif row['단위코드'] == '04': #
            x = 1000
        elif row['단위코드'] == '06': #
            x = 1 / 0.860
        elif row['단위코드'] == '08':#
            x = 1 / 3.6
        else:
            x = 63.4 * 1 / 3.6  # UNIT_CD = 14 #
        return x * row['사용량']

    df_en['USE_QTY_kWh'] = df_en.apply(lambda row: energy_conversion(row), axis=1)


    ## 에너지 종류에 따른 열 구분 ##########################################################################################
    def divide_energy(row):
        if row['에너지종류'] == '전기':
            x = [1, 0, 0]
        elif row['에너지종류'] == '도시가스':
            x = [0, 1, 0]
        else:
            x = [0, 0, 1] # 지역난방
        return np.array(x) * row['USE_QTY_kWh']
    df_en[['electricity_kWh', 'gas_kWh', 'districtheat_kWh']] = df_en.apply(lambda row: divide_energy(row), axis=1, result_type='expand')


    ## 연간 총 에너지 사용량 산출 ##########################################################################################
    df_1 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])[['USE_QTY_kWh','electricity_kWh','gas_kWh','districtheat_kWh']].sum() ## 연간 총 에너지사용량 산출
    df_2 = df_en.groupby(by = ['매칭표제부PK','에너지종류','year_use'])['USE_QTY_kWh'].count().to_frame() ## 매칭표제부PK, 에너지종류 및 연도별로 존재하는 갯수 산출
    df_2.columns = ['count_energy']
    df_en_annual = pd.concat([df_1, df_2], axis = 1).sort_index(axis=1) ## df_1, df_2 병합
    df_en_annual.reset_index(inplace=True)


    ## 전기E의 연간Data가 모두 존재하는 매칭표제부PK만 도출 ###################################################################
    df_result = pd.DataFrame()
    for t in df_en_annual['에너지종류'].unique():
        df_ = df_en_annual[df_en_annual['에너지종류']==t]
        if t == '전기':
            df_ = df_[df_['count_energy']%12==0]    ## 계량기가 두개이상일 경우 여기서 각각 개별적으로 연간데이터가 모두 존재할 때는 12로 나누어져야 하기 때문에 나머지가 0이 되어야 함.
        else:
            df_ = df_   # 전기를 제외한 나머지에 대해서는 굳이 연간 데이터 모두 존재하지 않아도 넘어감.
        df_result = pd.concat([df_result,df_])
    print('# of PK4 : ', df_result['매칭표제부PK'].nunique())    ## of PK4 : 병합(2768)/2018(2619)/2019(2666)/2020(2679)/2021(2746)/2022(2768)
    print('# of ROW4 : ',len(df_result))                       ## of ROW4 :  병합(4382)/2018(4060)/2019(4182)/2020(4211)/2021(4277)/2022(4382)


    ## 건물에서 전력소비는 필수이므로 전기데이터가 있는 PK만 체크################################################################
    # 앞서서는 전기의 연간 Data만 초점에 맞춰 사용하였기 때문에 PK 중 전기가 존재하는 PK만 추출 하기 위해 다음과 같은 작업을 진행
    electric_PKs = df_result[df_result['에너지종류'] == '전기']['매칭표제부PK'].unique()
    df_en_annual = df_result[df_result['매칭표제부PK'].isin(electric_PKs)]
    print('# of PK5 : ', df_en_annual['매칭표제부PK'].nunique())     ## of PK5 :  병합(2506)/2018(2296)/2019(2365)/2020(2349)/2021(2385)/2022(2506)
    print('# of ROW5 : ',len(df_en_annual))                        ## of ROW5 :  병합(4113)/2018(3728)/2019(3869)/2020(3869)/2021(3900)/2022(4113)


    ## excel Data로 export #############################################################################################
    df_en_annual.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\연간총에너지사용량("+A+").xlsx")
    return df_en_annual



# 2018~2022 병합 데이터
data_processing('병합')

# 2018~2022 중 개별 연도 입력 시 개별적 데이터 출력 가능
data_processing('2018')
data_processing('2019')
data_processing('2020')
data_processing('2021')
data_processing('2022')
