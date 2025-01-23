import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
import openpyxl
import xlsxwriter

file1 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2018년).xlsx", sheet_name="표제부_에너지사용량_계량기_2018년", dtype=str)

def tap(A,B):
    file7 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2022년).xlsx", sheet_name="에너지용도코드", dtype=str)
    # file7에서 "사용용도코드"와 "에너지종류" 열을 선택합니다.
    mapping_data = file7[['기관코드', '에너지종류']]
    # "사용용도코드"를 "용도코드"와 비교하여 동일한 경우 "에너지종류"로 수정합니다.
    A['에너지종류'] = A.apply(lambda row: mapping_data[mapping_data['기관코드'] == row['에너지공급기관코드']]['에너지종류'].values[0] if row['에너지공급기관코드'] in mapping_data['기관코드'].values else '', axis=1)
    file8 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(에너지사용량_2022년).xlsx", sheet_name="단위코드", dtype=str)
    mapping_data = file8[['단위코드', '단위명']]
    A['단위명'] = A.apply(lambda row: mapping_data[mapping_data['단위코드'] == row['단위코드']]['단위명'].values[0] if row['단위코드'] in mapping_data['단위코드'].values else '', axis=1)
    # A.to_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\[데이터 병합]표제부\병합데이터\[데이터 편집]표제부_"+str(B)+".xlsx", index=False)
    return A

file2=tap(file1, 2018)

file2['사용량'] = file2['사용량'].astype(float)
result = file2.groupby(['사용년월','에너지종류','단위명','매칭총괄표제부PK'])['사용량'].sum().reset_index()#'에너지공급기관코드',사용량일련번호,사용용도코드,'사용시작일자','사용종료일자' 날림
result.to_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]에너지사용량_2018.xlsx", index=False)
