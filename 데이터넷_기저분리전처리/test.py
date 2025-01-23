import os
import pandas as pd
base_dir1=r"C:\Users\user\Desktop\학부연구생 모음집\데이터넷 과제\기저분리데이터\데이터넷_기저분리_엑셀파일"
file01 = os.path.join(base_dir1,'데이터넷_아주대1__1_서울경기_2018_2019년_총괄표제부_계량기별_사용량_용도전체.csv')
file02 = os.path.join(base_dir1,'데이터넷_아주대1__2_서울경기_2018_2019년_총괄표제부_에너지원별_사용량_건물용도.csv')

file03=os.path.join(base_dir1,'데이터넷_아주대1__3_서울경기_2018_2019년_표제부_계량기별_사용량_용도전체.csv')
file04 =os.path.join(base_dir1,'데이터넷_아주대1__4_서울경기_2018_2019년_표제부_에너지원별_사용량_건물용도.csv')

file05=os.path.join(base_dir1,'데이터넷_아주대1__5_서울경기_의료기관_총괄표제부.csv')

file06=os.path.join(base_dir1,'데이터넷_아주대1__6_서울경기_의료기관_표제부.csv')
file07=os.path.join(base_dir1,'데이터넷_아주대1__7_서울경기_의료기관_층별개요.csv')

file01=pd.read_csv(file01, encoding='utf-8-sig')
file02=pd.read_csv(file02, encoding='utf-8-sig')
file03=pd.read_csv(file03, encoding='utf-8-sig')
file04=pd.read_csv(file04, encoding='utf-8-sig')
file05=pd.read_csv(file05, encoding='utf-8-sig')
file06=pd.read_csv(file06, encoding='utf-8-sig')
file07=pd.read_csv(file07, encoding='utf-8-sig')



def year_separation(A):
    A['use_ym'] = pd.to_datetime(A['use_ym'], format='%Y%m')
    A['year_use'] = A['use_ym'].dt.year           # 연도 구분
    A['month_use'] = A['use_ym'].dt.month         # 월별 구분은 추후에 진행할 예정
    return A

file01=year_separation(file01)
file02=year_separation(file02)
file03=year_separation(file03)
file04=year_separation(file04)
file05=year_separation(file05)
file06=year_separation(file06)
file07=year_separation(file07)


print(file03)



def year_division(A):
    B = A[A['use_ym']=='2018']
    C = A[A['use_ym']=='2019']
    return B,C
file01_2018,file01_2019=year_division(file01)
file02_2018,file02_2019=year_division(file02)
file03_2018,file03_2019=year_division(file03)
file04_2018,file04_2019=year_division(file04)
file05_2018,file05_2019=year_division(file05)
file06_2018,file06_2019=year_division(file06)
file07_2018,file07_2019=year_division(file07)

