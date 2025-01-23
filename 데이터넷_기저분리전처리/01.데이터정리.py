import os

import pandas as pd

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


columns_list = [set(df.columns) for df in [df01, df02, df03, df04, df05, df06, df07]]

# 모든 데이터프레임의 컬럼명의 교집합을 찾습니다.
common_columns = set.union(*columns_list)

# 교집합 컬럼명을 리스트 형태로 변환합니다.
common_columns_list = pd.DataFrame(common_columns)
print(common_columns_list)




# base_dir2=r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일"
#
#
# file_path_1= os.path.join(base_dir2, "데이터넷_아주대1__1_서울경기_2018_2019년_총괄표제부_계량기별_사용량_용도전체.csv")
# file_path_2= os.path.join(base_dir2, "데이터넷_아주대1__2_서울경기_2018_2019년_총괄표제부_에너지원별_사용량_건물용도.csv")
# file_path_3= os.path.join(base_dir2, "데이터넷_아주대1__3_서울경기_2018_2019년_표제부_계량기별_사용량_용도전체.csv")
# file_path_4= os.path.join(base_dir2, "데이터넷_아주대1__4_서울경기_2018_2019년_표제부_에너지원별_사용량_건물용도.csv")
# file_path_5= os.path.join(base_dir2, "데이터넷_아주대1__5_서울경기_의료기관_총괄표제부.csv")
# file_path_6= os.path.join(base_dir2, "데이터넷_아주대1__6_서울경기_의료기관_표제부.csv")
# file_path_7= os.path.join(base_dir2, "데이터넷_아주대1__7_서울경기_의료기관_층별개요.csv")
#
#
#
# file_path=os.path.join(base_dir2, "컬럼정리.csv")
# common_columns_list.to_csv(file_path, index=True, encoding='utf-8-sig')
# df01.to_csv(file_path_1, index=True, encoding='utf-8-sig')
# df02.to_csv(file_path_2, index=True, encoding='utf-8-sig')
# df03.to_csv(file_path_3, index=True, encoding='utf-8-sig')
# df04.to_csv(file_path_4, index=True, encoding='utf-8-sig')
# df05.to_csv(file_path_5, index=True, encoding='utf-8-sig')
# df06.to_csv(file_path_6, index=True, encoding='utf-8-sig')
# df07.to_csv(file_path_7, index=True, encoding='utf-8-sig')


################################################ 연도 구분 #############################################################
### 연도 월별 구분
def year_separation(A):
    A['use_ym'] = pd.to_datetime(A['use_ym'], format='%Y%m')
    A['year_use'] = A['use_ym'].dt.year           # 연도 구분
    A['month_use'] = A['use_ym'].dt.month         # 월별 구분은 추후에 진행할 예정
    return A

df01=year_separation(df01)
df02=year_separation(df02)
df03=year_separation(df03)
df04=year_separation(df04)
print(df01.head())


### 2018, 2019연도 구분
def year_division(A):
    B = A[A['year_use']==2018]
    C = A[A['year_use']==2019]
    return B,C
df01_2018,df01_2019=year_division(df01)
df02_2018,df02_2019=year_division(df02)
df03_2018,df03_2019=year_division(df03)
df04_2018,df04_2019=year_division(df04)



################################################ 연도별 Data 저장 #######################################################
def save_data_division(A):
    base_dir3 = os.path.join(r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일", str(A))
    file_path_1= os.path.join(base_dir3, "데이터넷_아주대1__1_서울경기_"+str(A)+"년_총괄표제부_계량기별_사용량_용도전체.csv")
    file_path_2= os.path.join(base_dir3, "데이터넷_아주대1__2_서울경기_"+str(A)+"년_총괄표제부_에너지원별_사용량_건물용도.csv")
    file_path_3= os.path.join(base_dir3, "데이터넷_아주대1__3_서울경기_"+str(A)+"년_표제부_계량기별_사용량_용도전체.csv")
    file_path_4= os.path.join(base_dir3, "데이터넷_아주대1__4_서울경기_"+str(A)+"년_표제부_에너지원별_사용량_건물용도.csv")

    eval(f'df01_{A}').to_csv(file_path_1, index=True, encoding='utf-8-sig')
    eval(f'df02_{A}').to_csv(file_path_2, index=True, encoding='utf-8-sig')
    eval(f'df03_{A}').to_csv(file_path_3, index=True, encoding='utf-8-sig')
    eval(f'df04_{A}').to_csv(file_path_4, index=True, encoding='utf-8-sig')

save_data_division(2018)
save_data_division(2019)


################################################ 연도별 Data 저장 #######################################################

A=2018
df_2 = eval(f'df01_{A}').groupby(by = ['mgm_bld_pk','use_purps_cd'])['month_use'].count().to_frame() ## 매칭표제부PK, 에너지종류 및 연도별로 존재하는 갯수 산출
