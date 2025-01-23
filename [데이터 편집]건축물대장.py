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


file1 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\데이터넷_의료시설(건축물대장,에너지사용량)\데이터넷_의료시설(건축물대장).xlsx", sheet_name="건축물대장(표제부)", dtype=str)
columns_to_remove = ['Unnamed: 78', 'Unnamed: 79', 'Unnamed: 80', '매칭총괄표제부PK']
file1 = file1[file1['매칭총괄표제부PK'].isnull()].drop(columns=columns_to_remove)
file1.to_excel(r'C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]건축물대장.xlsx', index=False)#행개수 4318 ea

# result = file1.groupby(['매칭총괄표제부PK', '사용년월'])['사용량'].sum().reset_index()