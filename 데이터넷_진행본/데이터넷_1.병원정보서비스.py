import os
import pandas as pd

# 데이터셋 2 Data Import ################################################################################################
## 병원정보서비스
dir_path = r"C:\Users\user\Desktop\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결"
filename = "1.병원정보서비스 2022.10..csv"
file_path = os.path.join(dir_path,filename)
hos_00 = pd.read_csv(file_path)
print('# of data : ',len(hos_00)) ##행 갯수 :  76032

## 시설정보
filename = "3.의료기관별상세정보서비스_01_시설정보_202309.csv"
file_path = os.path.join(dir_path,filename)
hos_01 = pd.read_csv(file_path, encoding='cp949')
print('# of data : ',len(hos_01)) ##행 갯수 :  100335



# 1.병원정보서비스 기준 Case2 데이터 전처리 #################################################################################
def Data_Programming(A):
    hos = A[A['mgm_bld_pk'].notna()] ## 표제부!=NaN
    hos = hos[hos['mgm_upper_bld_pk'].isna()] ## 총괄표제부==NaN
    base_dir = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"

    ## 데이터넷_1.건축물대장.py에서 전처리된 건축물대장 excel 입력
    filenames_bldg="데이터넷_건축물대장(일반).xlsx"
    file_path = os.path.join(base_dir,filenames_bldg)
    df_bldg2 = pd.read_excel(file_path)

    PK_list = df_bldg2['매칭표제부PK'].unique()  ## 전처리된 건축물대장 PK list입력
    hos = hos[hos['mgm_bld_pk'].isin(PK_list)]

    hos['PK당요양기호'] = hos.groupby(['mgm_bld_pk'])['암호화요양기호'].transform('count') ## 매칭표제부 당 암호화요양기호 갯수 산정
    hos = hos[hos['PK당요양기호'] == 1].copy() ## 표제부당 암호화요양기호가 1개인 것만 추출
    hos.drop(columns={'PK당요양기호'}, inplace=True) ## PK당요양기호 열 제거
    return hos

hos_00=Data_Programming(hos_00) ## 병원정보서비스_Data
hos_01=Data_Programming(hos_01) ## 시설정보_Data
print('# of data1 : ',len(hos_00)) ## of data1 :  28107
print('# of data2 : ',len(hos_01)) ## of data2 :  29363


hos_merge = pd.merge(left=hos_00, right=hos_01, how='inner', on=['암호화요양기호','mgm_bld_pk','종별코드','종별코드명']) ## 암호화요양기호 및 매칭표제부 외 2개 기준 병합
print('# of data3 : ',len(hos_merge)) ## of data :  22642


hos_merge = hos_merge[hos_merge['종별코드명'].isin(['병원', '치과병원','한방병원','요양병원','정신병원','종합병원'])] ##대상 병원 필터
print('# of data4 : ',len(hos_merge))# of data :  2153


hos_merge.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\test.xlsx")




# 건축물 대장+에너지사용량(2022) 병합본 반영 #################################################################################
def year_data(A):
    print("#####################################")
    file_path1=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\대장에너지병합(" + A + ").xlsx"
    df_bldg2 = pd.read_excel(file_path1)
    hos_merge1 = pd.merge(left=df_bldg2, right=hos_merge, how='inner',left_on=['매칭표제부PK'], right_on=['mgm_bld_pk'])
    print("# of PK : ", hos_merge1['매칭표제부PK'].nunique())# of PK :  2018(1273)/2019(1356)/2020(1359)/2021(1364)/2022(1456)
    print('# of data4 : ',len(hos_merge1))                 # of data :  2018(2221)/2019(2386)/2020(2414)/2021(2404)/2022(2585)
    hos_merge1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\최종결합본(" + A + ").xlsx")
    return hos_merge1
year_data('2018')
