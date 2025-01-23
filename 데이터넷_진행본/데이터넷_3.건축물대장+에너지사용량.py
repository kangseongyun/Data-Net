import pandas as pd

def data_merge(A):
    ## 필터_건축물대장(일반)###############################################################################################
    file_path1=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_건축물대장(최종).xlsx"
    df_bldg2 = pd.read_excel(file_path1)
    print("# of PK : ", df_bldg2['매칭표제부PK'].nunique())# of PK :  3758
    print('# of data : ', df_bldg2.shape[0])# of data :  3758

    ## 필터_연간총에너지사용량##############################################################################################
    file_path=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\연간총에너지사용량("+A+").xlsx"
    df_en_annual = pd.read_excel(file_path)
    print("# of PK : ", df_en_annual['매칭표제부PK'].nunique())# of PK :  2018(2911)/2019(3014)/2020(3010)/2021(3086)/2022(3249)
    print('# of data : ', df_en_annual.shape[0])             # of data : 2018(4712)/2019(4923)/2020(4952)/2021(5040)/2022(5321)


    ## 필터_건축물대장(일반)+필터_연간총에너지사용량>>>>>Data Merging###########################################################
    df_combined1 = pd.merge(left=df_en_annual, right=df_bldg2, how='left', on ='매칭표제부PK')
    print("# of PK : ", df_combined1['매칭표제부PK'].nunique())# of PK :  2018(2911)/2019(3014)/2020(3010)/2021(3086)/2022(3249)
    print('# of data : ', df_combined1.shape[0])            # of data :  2018(2911)/2019(3014)/2020(3010)/2021(3086)/2022(3249)
    df_combined1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\대장에너지병합("+A+").xlsx")
    return df_combined1

data_merge('2018')
data_merge('2019')
data_merge('2020')
data_merge('2021')
data_merge('2022')